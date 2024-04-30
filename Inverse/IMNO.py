
import torch
import os
import argparse
from matplotlib.lines import Line2D
from neuralop.models import FNO
from torch import nn
import json
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
class Branch_net(nn.Module):
    def __init__(self,input,hidden,output):
        super(Branch_net,self).__init__()
            
        self._net = nn.Sequential(
                     nn.Linear(input, hidden),
                     nn.Tanh(),
                     nn.Linear(hidden, hidden),
                     nn.Tanh(),
                     nn.Linear(hidden, output),
                     nn.Tanh(),
                    )
    def forward(self,x):
        out=self._net(x)
        return out
     
def to_numpy(tensor):
    """Ensure tensor is detached and moved to CPU if needed before converting to NumPy array."""
    return tensor.detach().cpu().numpy() if tensor.is_cuda else tensor.detach().numpy()


class MFNO(nn.Module):
  
  def __init__(self,infeature=2,Trunk="FNO",t_steps=640,wavelet="db6",fno_modes=64):
      super(MFNO,self).__init__()
      self.bran_nn = Branch_net(infeature,50,1)
      self.Trunk = Trunk
      if self.Trunk  == "FNO":
        #modes64 的参数量 2605052
        print("FNO")
        self.trunk_nn = FNO(n_modes=(fno_modes,fno_modes),hidden_channels=12,in_channels=1,out_channels=t_steps)
        
      elif self.Trunk  == "UNO":
        print("UNO")
        #modes64 的参数量 2656746
        self.trunk_nn = UNO(1,640, hidden_channels=10, projection_channels=12,uno_out_channels = [8,16,16,16,8], \
            uno_n_modes= [[fno_modes,fno_modes],[32,32],[32,32],[32,32],[fno_modes,fno_modes]], uno_scalings=  [[1.0,1.0],[0.5,0.5],[1,1],[2,2],[1,1]],\
            horizontal_skips_map = None, n_layers = 5, domain_padding = 1)
        
      elif self.Trunk  == "WNO":
        h = 2 # dwt需要偶数 
        s = 300
        print("wno")
      
        layers = 6 #level = 2 , 2606035 参数量 fno modes为分解的level：1/2/3/4 out is tsteps
        self.trunk_nn = WNO2d(width= 6, level = fno_modes, layers = layers , size=[h,s], wavelet= wavelet,
              in_channel = 3, grid_range=[1,1], padding=0,out=640).to(device) #实际输出【1，1，3，640】
        #输入[1, 1,1, 300]输出[1,640,1,300]
        
      
      self.condition_norm = nn.BatchNorm1d(infeature)  # 归一化层
      self.ini_norm = nn.BatchNorm2d(1)

  def forward(self,x,condition):
      #对序列进行fno
      # input:[batch,1,1,300], irst 1 is time
      x_norm =self.ini_norm(x)
      condition_normal= self.condition_norm(condition)
      #norm
      if (self.Trunk  == "WNO"):
        #wno输入应该是【b，1，300，1】#time 放最后
        x_norm = x_norm.permute(0,2,3,1)
        trunk_out = self.trunk_nn(x_norm)
        #note： wno 去掉一个维度,因为输入得是偶数
        trunk_out = trunk_out[:,0:1,:,:] #[b,1,300.640]
        trunk_out =trunk_out.permute(0,3,1,2)#out [b,640,3,300]
      else:
        
         trunk_out = self.trunk_nn(x_norm) #[b, 640, 3, 300]，640是时间步
        
      
      out = self.bran_nn(condition_normal)
  
      out = out.unsqueeze(-1)
      out = out.unsqueeze(-1)
      #([batch, 640, 1, 300])
      final_out = trunk_out*out
   
      return final_out
class Expr_Inverse():
   
   def __init__(self,
                device_set,
                save_name,
                config:dict,
                net,
                ini,
                eta,
                true_x, 
                **kwargs):
  

      self.decoder = kwargs.get("norm_decoder", None)
      self.decoder.cuda()
      net_dict = torch.load(net)#map_location=torch.device('cpu')
      self.net = MFNO()
      self.net.load_state_dict(net_dict)
      self.net =self.net.to(device_set)
      self.save_name = save_name
      #######
      set_seed(config["seed"])

      self.branch_net = self.net.bran_nn
      self.trunk_net = self.net.trunk_nn
      self.config = config

      # self.trunk_net 是你想要冻结的网络部分
      for param in self.trunk_net.parameters():
         param.requires_grad = False

           
      self.eta = eta #[1,640,1,300]
      self.ini = ini #[1,1,1,300]

      self.ini_truc = self.trunk_net(self.ini)

      self.true_x =  true_x


      #  核心
      self.num_particles = self.config["num_particles"]
      dimensions = self.config["dimensions"]
      self.bounds = self.config["bound"] # [(0,0.1),(1,5)]
      #为0 
      self.x = torch.zeros(self.num_particles, dimensions).to(device_set)
      self.v = torch.zeros(self.num_particles, dimensions).to(device_set)
      self.mse_loss = nn.MSELoss(reduction='none')
      
        
      #初始化位置
      for dim in range(dimensions):
        self.x[:, dim] = torch.rand(self.num_particles) * (self.bounds[dim][1] - self.bounds[dim][0]) + self.bounds[dim][0]


      self.p_best = self.x.clone()
      self.g_best = self.x[self.PIMNO_function(self.x,self.true_x).argmin()] #

   
      # 初始化历史记录列表
      self.positions_history = [self.x]
      self.objective_history = [self.PIMNO_function(self.x,self.true_x)]
   
   
   def PIMNO_function(self, x, true_x,**kwargs):


      
      self.net.eval() #评估，在评估模式下，BatchNorm 层会使用训练时累积的运行均值和方差，而不是当前批次的统计数据
      num_particle = x.shape[0]
      true_x_broadcasted = true_x.expand(x.size(0), -1)
      x.requires_grad_(True) 
      self.hat_cb = self.branch_net(x)
      grad_outputs = torch.ones_like(self.hat_cb)

      s_grads = torch.autograd.grad(outputs=self.hat_cb, inputs=x, grad_outputs=grad_outputs,
                                    create_graph=True)[0]

      x_detached = x.detach().clone()
      x_detached[:, 0] = true_x_broadcasted[:, 0]
      x_detached.requires_grad_(True)
      self.true_a_cb = self.branch_net(x_detached)
      t_grad = torch.autograd.grad(outputs=self.true_a_cb, inputs=x_detached, grad_outputs=grad_outputs,
                                    create_graph=True)[0]

      g1_loss = self.mse_loss(s_grads, t_grad).mean(1)
      yc1_loss = self.mse_loss(true_x_broadcasted[:, 0], x[:, 0])

      self.eta_expand = self.eta.expand(num_particle, -1, -1, -1)
      self.ini_expand = self.ini.expand(num_particle,-1,-1,-1)

      if self.decoder != None:
         #反归一化
         print("norm")
         self.hat_eta = self.decoder.decode(self.net(self.ini, x) )#[b,640,1,300]
      else:

         self.hat_eta = self.net(self.ini, x) #[b,640,1,300]

      mse_eta = self.mse_loss(self.hat_eta, self.eta_expand).mean([1, 2, 3])
   
      del x
              # 清理内存
      del self.hat_eta, self.eta_expand

      torch.cuda.empty_cache()
      
      # 清除梯度
      for param in self.branch_net.parameters():
         param.grad = None
      
      # 计算总损失

      f = 1e8*mse_eta + self.config['gamma'] * g1_loss + self.config['lambda'] * yc1_loss 
      del x_detached,self.true_a_cb 
      return f


   def search(self,dat):
      
      #print(f"Function name: {func.__name__}")
      num_particles = self.config["num_particles"]
      dimension = self.config["num_particles"] #2
      
      self.optimize(
                   num_iterations= self.config["iterations"],
                   )
      self.save(dat=dat)
      
   def save(self,dat="test"):
      '''
      save to npz
      '''
        # 将字典保存为JSON文件
      with open(f'Innverse_Out/{dat}/args.json', 'w') as json_file:
            json.dump(self.config, json_file, indent=4)
      # Convert lists of tensor positions and objectives to numpy arrays
      positions_np = [to_numpy(pos) for pos in self.positions_history]  # Ensure tensors are moved to cpu and converted to numpy
      objectives_np = [to_numpy(obj) for obj in self.objective_history]

      # Save the arrays to an npz file
      np.savez_compressed(f"{self.save_name}.npz",
                           positions = positions_np, 
                          objectives = objectives_np,
                          gbest = to_numpy(self.g_best))

      print(f"Data saved to {self.save_name}.npz")
   
   def optimize(self, num_iterations, c1=0.03, c2=0.02, w=0.02):

      step_size = 0.5  # Define the step size for the second dimension

      for i in range(num_iterations):
         r1 = torch.rand_like(self.x)
         r2 = torch.rand_like(self.x)
         self.v = w * self.v + c1 * r1 * (self.p_best - self.x) + c2 * r2 * (self.g_best - self.x)

         self.x = self.x + self.v


         # Quantize the second dimension
         self.x[:, 1] = torch.round(self.x[:, 1] / step_size) * step_size

         # Apply boundary conditions for all dimensions
         for dim in range(self.x.shape[1]):
            self.x[:, dim] = torch.clamp(self.x[:, dim], min=self.bounds[dim][0], max=self.bounds[dim][1])

         # Store positions history
         self.positions_history.append(to_numpy(self.x))

         print("t1", self.x[:, 1])

         # Evaluate objective function
         objective_values = self.PIMNO_function(self.x, self.true_x)
         better_mask = objective_values < self.PIMNO_function(self.p_best, self.true_x)

         # Broadcast better_mask to appropriate dimensions
         better_mask = better_mask.unsqueeze(-1).expand_as(self.p_best)

         # Update personal best positions
         self.p_best = torch.where(better_mask, self.x, self.p_best)

         # Ensure p_best also respects boundary conditions
         for dim in range(self.p_best.shape[1]):
            self.p_best[:, dim] = torch.clamp(self.p_best[:, dim], min=self.bounds[dim][0], max=self.bounds[dim][1])


         # 更新全局最佳位置
         current_best_idx = objective_values.argmin()
         current_best_value = objective_values[current_best_idx]
         self.g_best = self.g_best.detach().reshape(-1,2)
         print("best",self.g_best)
      
         if (self.PIMNO_function(self.g_best, self.true_x) > current_best_value):#目标要更新
               
            self.g_best = self.x[current_best_idx]
         

         # 记录当前位置和目标函数值

         self.objective_history.append(objective_values.detach())
         #update 图
         self.update(iteration=i)
         print(f"step {i}: {objective_values.min()}")  # 数量应与粒子数相同
         print("best",self.g_best)
         torch.cuda.empty_cache()
         del r1,r2
      return self.g_best

   def update(self,iteration):
      plt.ion()  # Turn on interactive mode
      fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # Three subplots for three different visualizations
      clear_output(wait=True)
      # Generate a unique color for each star and create a colormap for each
      num_examples = 5
      colors = plt.cm.viridis(np.linspace(0, 1, num_examples))  # Use a color map to get distinct colors

      def to_numpy(obj):
      # 首先检查obj是否为PyTorch Tensor
         if isinstance(obj, torch.Tensor):
            # 检查张量是否在CUDA上
            if obj.is_cuda:
                  return obj.detach().cpu().numpy()  # 先从计算图中分离，然后移到CPU，最后转换为NumPy数组
            else:
                  return obj.detach().numpy()  # 如果在CPU上，直接分离并转换为NumPy数组
         elif isinstance(obj, np.ndarray):
            # 如果已经是NumPy数组，直接返回
            return obj

      ax[0].cla()
      ax[0].plot([to_numpy(o.min())  for o in self.objective_history], 'b-',label="Min of values")
      ax[0].set_title('Minimum F Value with Iterations',fontsize=20)
      ax[0].set_xlabel('Iteration',fontsize=16)
      ax[0].set_ylabel('F Value',fontsize=16)
      ax[0].set_yscale("log")
      ax[0].legend(loc='upper left')

      ax[1].cla()

      for idx,j in enumerate(range(num_examples)):
         traj = np.array([to_numpy(pos[j])for pos in self.positions_history])
         color = colors[idx]  # Pick color for this star
   
         ax[1].scatter(traj[:, 0], traj[:, 1], facecolors='none', 
                       edgecolors = color, s=(iteration+75)*1.25, 
                       marker='o', alpha=0.9,label=f"Particle_{idx}")  # Draw hollow circles
         ax[1].scatter(traj[:, 0], traj[:, 1], color=color, s=(iteration+50) * 1.15, marker='+', alpha=0.7)  # Draw small pluses
         


      
      ax[1].scatter(to_numpy(self.true_x[0,0]),to_numpy(self.true_x[0,1]), 
                    c='black', marker='*',s=150,label="True_Value")

      ax[1].set_title('Particle Trajectories',fontsize=20)
      ax[1].set_xlabel('A',fontsize=16)
      ax[1].set_ylabel('L',fontsize=16)
      ax[1].legend( loc='upper left')
      
      ax[2].cla()
      best_traj = np.array([to_numpy(pos[self.objective_history[k].argmin().item()]) 
                            for k, pos in enumerate(self.positions_history)])
     
      ax[2].plot(best_traj[:, 0], best_traj[:, 1], linestyle='--',  marker='>',color='m')
      ax[2].scatter(best_traj[:, 0], best_traj[:, 1], marker='+',
                    s= (iteration+70)*1.2, alpha=0.9,label="Best")
      ax[2].legend(loc='best')
      ax[2].set_title('Trajectory of the Best Particle',fontsize=20)
      ax[2].set_xlabel('A',fontsize=16)
      ax[2].set_ylabel('L',fontsize=16)

      plt.tight_layout()
      plt.savefig("trac.png",dpi=300)
      plt.close()

def args_to_config(args):
   '''
   args to config
   '''
   config = {
      'seed': args.seed,
      'gamma': args.gamma,
      'lambda': args.lamda,
      'bound': args.bound,
      'num_particles': args.num_particles,
      'dimensions': args.dimensions,
      'iterations':args.iterations
   }
   return config
  
def parse_bounds(bound_str):
    # 分解多个边界对，每个边界对由逗号分隔
    bound_pairs = bound_str.split(',')
    # 对每个边界对进一步分解并转换为浮点数，最后生成嵌套列表
    return [[float(bound) for bound in pair.split('^')] for pair in bound_pairs]

def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
if __name__ == "__main__":
  print("当前工作目录:", os.getcwd(),flush=True)
  # 创建 ArgumentParser 对象
  parser = argparse.ArgumentParser(description="Process Innverse.")
  # 添加参数
  parser.add_argument('--seed', type = int, 
                      default = 1234,
                      help = 'seeds exprs')
  parser.add_argument('--gamma',type = float, 
                      default=100000,
                      help = 'Gamma is in the yc1 gradients')
  parser.add_argument('--lamda',type = float, 
                      default=0,
                      help = 'Lambda is in the yc1')
  parser.add_argument('--bound',type=parse_bounds,
                      default="3^10,1^10",
                      help="bounds of dimensions")
  parser.add_argument("--num_particles",type = int,
                      default=1000,
                      help = "numbers")
  parser.add_argument("--dimensions",type = int,
                      default = 2,
                      help="yc1 and yc2")
  parser.add_argument("--iterations",type = int,
                      default = 10000,
                      help="iterations")
   # pareser.add_argument("--c1",type=float,
   #                      )
  #转移到dict
  #解析命令行参数
  args = parser.parse_args()
  config = args_to_config(args)
  #net 文件
  net_path = "Inverse/mno_ckpt.pth"
  set_seed(12)
  device_set ="cuda"#cpu or cuda
  ini = 2+torch.randn(1,1,1,300).to(device_set) # first 1 is time 
  eta = 2+torch.randn(1,640,1,300).to(device_set) 
  true_x = 2+ torch.randn(1,2).to(device_set) 
  print("true_x",true_x)

  expr = Expr_Inverse(config = config,
                      device_set = device_set,
                      save_name = "test",
                      net = net_path,
                      ini = ini,
                      eta = eta,
                      true_x=true_x,
                    
                       )
  expr.search(dat="test")
  
  
  
