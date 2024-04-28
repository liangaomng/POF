
import torch
import os
import argparse
from neuralop.models import FNO
from torch import nn

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
                save_name,
                config:dict,
                net,
                ini,
                eta,
                true_x):
  
      net_dict = torch.load(net)#map_location=torch.device('cpu')
      self.net = MFNO()
      self.net.load_state_dict(net_dict)
      self.net =self.net.to("cuda")
      #######
    
      self.branch_net = self.net.bran_nn
      self.trunk_net = self.net.trunk_nn
      self.config = config

           
      self.eta = eta #[1,640,1,300]
      self.ini = ini #[1,1,1,300]

      self.ini_truc = self.trunk_net(self.ini)

      self.true_x =  true_x


      #  核心
      self.num_particles = self.config["num_particles"]
      dimensions = self.config["dimensions"]
      self.bounds = self.config["bound"] # [(0,0.1),(1,5)]
      #为0 
      self.x = torch.empty(self.num_particles, dimensions).to("cuda")
      self.v = torch.zeros(self.num_particles, dimensions).to("cuda")
      self.mse_loss = nn.MSELoss(reduction='none')
      
        
      # 初始化位置
      for dim in range(dimensions):
         self.x[:, dim] = torch.rand(self.num_particles) * (self.bounds[dim][1] - self.bounds[dim][0]) + self.bounds[dim][0]

      self.p_best = self.x.clone()
      self.g_best = self.x[self.PIMNO_function(self.x,self.true_x).argmin()].clone() #

      # 初始化历史记录列表
      self.positions_history = [self.x.clone()]
      self.objective_history = [self.PIMNO_function(self.x,self.true_x).clone()]
   
   
   def PIMNO_function(self, x, true_x):
    # 确保维度正确
    #x   是条件
    num_particle = x.shape[0]
    print("num_particle:",num_particle)
    assert x.shape[1] == 2  # 确保每个粒子有两个维度

    # 自动广播 true_x 到每个粒子
    true_x_broadcasted = true_x.expand(x.size(0), -1)
    true_x_broadcasted.requires_grad_(True)
    x.requires_grad_(True)

    # 计算分支网络的输出
    self.hat_cb = self.branch_net(x) #[100,1]
    grad_outputs = torch.ones_like(self.hat_cb)

    # 计算关于x的梯度
    s_grads = torch.autograd.grad(outputs=self.hat_cb,
                                  inputs=x,
                                  grad_outputs=grad_outputs,
                                  create_graph=True)[0]

    # 使用detach防止梯度追踪
    x_detached = x.detach()

    # 替换第一维度的值为真实值
    x_detached[:, 0] = true_x_broadcasted[:, 0]
    x_detached.requires_grad_(True)

    # 计算分支网络的输出
    self.true_a_cb = self.branch_net(x_detached)
    t_grad = torch.autograd.grad(outputs=self.true_a_cb,
                                 inputs=x_detached,
                                 grad_outputs=grad_outputs,
                                 create_graph=True)[0]

    # 计算梯度损失和位置损失
    g1_loss = self.mse_loss(s_grads, t_grad).mean(1)
    yc1_loss = self.mse_loss(true_x_broadcasted[:, 0], x[:, 0])

    # 扩展 hat_cb 以匹配 eta 的形状,有些时候需要1，因为是best的时候
    self.ini_truc =self.trunk_net(self.ini)
    self.ini_truc = self.ini_truc.expand(num_particle,self.ini_truc.shape[1],self.ini_truc.shape[2],self.ini_truc.shape[3])
    self.eta_expand = self.eta.expand(num_particle,self.ini_truc.shape[1],self.ini_truc.shape[2],self.ini_truc.shape[3])

    #hat eta
    self.hat_eta = self.branch_net(x).unsqueeze(-1) .unsqueeze(-1) * self.ini_truc

    mse_eta = self.mse_loss(self.hat_eta, self.eta_expand).mean([1, 2, 3]) 

   
     # 计算每个粒子的总损失
    f = mse_eta +  self.config["gamma"] * g1_loss + self.config["lambda"] * yc1_loss
    del  self.hat_eta
    del  self.eta_expand
    torch.cuda.empty_cache()  # Free up any released memory immediately

    return f  # f 应该是 [num_particles] 的张量

      
   def search(self):
      
      #print(f"Function name: {func.__name__}")
      num_particles = self.config["num_particles"]
      dimension = self.config["num_particles"] #2
      
      self.optimize(
                   num_iterations= self.config["iterations"],
                   )
   
   def optimize(self, num_iterations, c1=1.5, c2=1.5, w=0.9):
      with torch.no_grad():  # 在不需要计算梯度的环境中执行
         for i in range(num_iterations):
            r1 = torch.rand_like(self.x)
            r2 = torch.rand_like(self.x)
            self.v = w * self.v + c1 * r1 * (self.p_best - self.x) + c2 * r2 * (self.g_best - self.x)
            self.x = self.x + self.v


            # 更新个体最佳位置和全局最佳位置
            # 应用边界条件
            for dim in range(self.x.shape[1]):
               self.x[:, dim] = torch.clamp(self.x[:, dim], min=self.bounds[dim][0], max=self.bounds[dim][1])
               print("___x")
         

            
            objective_values = self.PIMNO_function(self.x, self.true_x)  
            better_mask = objective_values < self.PIMNO_function(self.p_best, self.true_x)
            # 使用 better_mask 广播到合适的维度
            better_mask = better_mask.unsqueeze(-1).expand_as(self.p_best)
            self.p_best = torch.where(better_mask, self.x.detach(), self.p_best)

            #update 图
            #self.update()

            # 更新全局最佳位置
            current_best_idx = objective_values.argmin()
            current_best_value = objective_values[current_best_idx]
            self.g_best = self.g_best.detach().reshape(-1,2)
         
            if (self.PIMNO_function(self.g_best, self.true_x) < current_best_value).all():#all 表示tensor里面所有的值都小
                  
               self.g_best = self.x[current_best_idx].detach().clone()
            

            # 记录当前位置和目标函数值
         
            self.positions_history.append(self.x.clone())
            self.objective_history.append(objective_values.clone())
            print(f"step {i}: {objective_values.min()}")  # 数量应与粒子数相同
            print("best",self.g_best)
            torch.cuda.empty_cache()
         return self.g_best

   def update(self):
      plt.ion()  # Turn on interactive mode
      fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # Three subplots for three different visualizations
      clear_output(wait=True)
      
      ax[0].cla()
      ax[0].plot([o.min().item() for o in self.objective_history], 'r-')
      ax[0].set_title('Minimum F Function Value Over Iterations')
      ax[0].set_xlabel('Iteration')
      ax[0].set_ylabel('Minimum F Value')

      ax[1].cla()
      num_examples = 10
      for j in range(num_examples):
         traj = np.array([pos[j].detach().numpy() for pos in self.positions_history])
         ax[1].plot(traj[:, 0], traj[:, 1], marker='o', linestyle='-')
      ax[1].set_title('Example Particle Trajectories')
      ax[1].set_xlabel('Dimension 1')
      ax[1].set_ylabel('Dimension 2')

      ax[2].cla()
      best_traj = np.array([
         pos[self.objective_history[k].argmin().detach().item()].detach().numpy()  # Ensure that tensors are detached before converting to numpy
         for k, pos in enumerate(self.positions_history)
      ])
      ax[2].plot(best_traj[:, 0], best_traj[:, 1], marker='o', linestyle='-', color='m')
      ax[2].set_title('Trajectory of the Best Particle')
      ax[2].set_xlabel('Dimension 1')
      ax[2].set_ylabel('Dimension 2')
      plt.savefig("trac.png",dpi=300)
      plt.pause(0.05)  # Pause to update plots

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
                      default=0,
                      help = 'Gamma is in the yc1 gradients')
  parser.add_argument('--lamda',type = float, 
                      default=0,
                      help = 'Lambda is in the yc1')
  parser.add_argument('--bound',type=parse_bounds,
                      default="3^10,1^10",
                      help="bounds of dimensions")
  parser.add_argument("--num_particles",type = int,
                      default=100,
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
  
  ini = 2+torch.randn(1,1,1,300).to("cuda") # first 1 is time 
  eta = 2+torch.randn(1,640,1,300).to("cuda") 
  true_x =2+ torch.randn(1,2).to("cuda") 
  print("true_x",true_x)

  expr = Expr_Inverse(config = config,
                      save_name = "test",
                      net = net_path,
                      ini = ini,
                      eta = eta,
                      true_x=true_x)
  expr.search()
  
  
  
