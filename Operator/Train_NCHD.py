
from neuralop.models import FNO,UNO
from torch import nn
from Dataset_torch import Read_Mat_4torch
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from Data_clean import return_name_list
from Evaluate import global_metric,local_metric
import argparse
import json
import shutil
import utilities3 
from tqdm import tqdm  # Correct import
from Dataset_torch import OE_Dataset,Get_4_folder
from wavelet_convolution import WaveConv2d
import torch.nn.functional as F
class WNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, padding=0,out=1):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
        
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 2 elements (for 2D), image size
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range( self.layers ):
            self.conv.append( WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet) )
            self.w.append( nn.Conv2d(self.width, self.width, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
                
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]     
        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        x = F.gelu( self.fc1(x) )            # Shape: Batch * x * y * Channel
        x = self.fc2(x)                      # Shape: Batch * x * y * Channel
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
# from Evaluate import calculate_psnr,calculate_ssim
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

class PID_alpha():
  
  def __init__(self,kp=None,ki=None,kd=None,*,alpha,pid_en): 
    
    self.pid_en = pid_en 
    if pid_en ==True:
        self.kp = kp if kp is not None else 0.1
        self.ki = ki if ki is not None else 0.01
        self.kd = kd if kd is not None else 0.01
    self.alpha = alpha
    self.integral = 0  # 积分项累计
    self.previous_error = 0  # 存储前一次误差，用于计算微分项
    

      
  def __call__(self,error):
    
    if self.pid_en ==True:
      
        self.integral += error
        self.derivative = error - self.previous_error
        
        self.alpha +=  self.kp * error + self.ki * self.integral + self.kd * self.derivative
        
        # 限制alpha值在0.05到0.95之间  
        self.alpha = max(0.1, min(0.9, self.alpha))
        self.previous_error =error #  更新误差
    else:
      
      self.alpha = self.alpha
    
    return self.alpha

class MFNO(nn.Module):
  
  def __init__(self,infeature=2,Trunk="FNO",t_steps=640):
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
        self.trunk_nn = WNO2d(width= 14, level = fno_modes, layers = layers , size=[h,s], wavelet='db6',
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
  def stat_trunk_params(self):
       # 遍历模型参数
      self.total_params=0
      for param in self.trunk_nn.parameters():
          self.total_params += param.numel()
      return self.total_params
      
    
       
def plot_test_performance(folder,epoch,**kwargs):
    #SSIM是基于对亮度、对比度和结构三个不同维度的比较计算得出的。
    # 具体来说，SSIM越接近1，意味着两幅图像在视觉上越相似；
    # SSIM越接近0，意味着两幅图像在视觉上的差异越大。
    
    g_SSIM = kwargs["Global_metric"].Global_dict["ssim"]
    g_PSNR = kwargs["Global_metric"].Global_dict["psnr"]
    g_MAPE = kwargs["Global_metric"].Global_dict["mape"]
    
    #local
    l_SSIM = kwargs["Local_metric"].Local_dict["ssim"]
    l_PSNR = kwargs["Local_metric"].Local_dict["psnr"]
    l_MAPE = kwargs["Local_metric"].Local_dict["mape"]
    
    #loss

    test_losses = kwargs["test_losses"]
    
    num_points= int (epoch / args.save_epoch) + 1
      
    fig,ax= plt.subplots(4,1,figsize=(14,12))
    
    ax[0].plot(range(num_points),g_PSNR, label="Global PSNR", linestyle='-',
               linewidth=3,marker="o")
    ax[0].plot(range(num_points),l_PSNR, label="Local PSNR", linestyle='--',
               linewidth=3,marker="s")
  
    # 设置刻度标签，确保标签反映实际的epoch数
    ax[0].set_title(f"PSNR_{epoch}_"+kwargs["title"]+f"_alpha_{pid_alpha.alpha}")
    ax[0].legend()
    ax[0].set_xlabel(f'x{args.save_epoch} epoch ')  # 添加x轴的标签
    
    ax[1].plot(range(num_points),g_SSIM,label="Global SSIM",linestyle='-',
               linewidth=3,marker="o")
    ax[1].plot(range(num_points),l_SSIM,label="Local SSIM",linestyle='--',
               linewidth=3,marker="s")
    ax[1].set_title(f"SSIM_{epoch}_"+kwargs["title"])
    ax[1].legend()
    
    ax[2].plot(range(num_points),g_MAPE, label="Global MAPE",linestyle='-',
               linewidth=3,marker="o")
    ax[2].plot(range(num_points),l_MAPE, label="Local MAPE",linestyle='--',
               linewidth=3,marker="s")
    ax[2].set_yscale('log')
    ax[2].set_title(f"MAPE_{epoch}_"+kwargs["title"])
    ax[2].legend()
    

    ax[3].plot(range(num_points),test_losses, label="Test loss",linewidth=3,
             linestyle='--')
    ax[3].legend()
    ax[3].set_yscale('log')
    ax[3].set_title(f"Losses up to Epoch")
    
    plt.tight_layout()
    
    plt.savefig(f"{folder}/Test_performance_{kwargs['title']}.png")
    plt.close()
   
# 测试 and plot 
def test(folder,epoch,test_loader,device):

    with torch.no_grad():
      mse=torch.nn.MSELoss()
      loss=0
      for i,(ini_data,condition,ground_truth) in enumerate(test_loader):
      
        condition = condition.to(device)
        
        ini_data = ini_data.to(device)
        ground_truth =ground_truth.to(device)
        

        #对整个序列做变换
        ini_data =ini_data .permute(0,2,1,3)#变成[20,1,3,300】 

        final_out= mNO (ini_data,condition) #[20,3,640,300]
        loss = mse (final_out[:,:,:,:],ground_truth[:,:,:,:]) # 归一化前算一个loss
        
        # 反归一化
        final_out =  gt_normalizer.decode(final_out)
        ground_truth= gt_normalizer.decode(ground_truth)
      
        
        pred_data =  final_out
        
        

    print("gt",ground_truth.shape) #[b,640,1,300]
    Mean_step_loss= loss.item()
    fig,ax= plt.subplots(1,3,figsize=(12,8))
    #画图预测第一个batch表示eta
    cax1=ax[0].imshow(ground_truth[0,:,0,:].cpu().numpy(),cmap="jet",vmin=0,vmax=0.05)
  
    ax[0].set_title("True"+f"A_{condition[0,0].item():.2f}"+f"_L_{condition[0,1].item():.2f}") 
    cax2=ax[1].imshow(pred_data[0,:,0,:].cpu().numpy(),cmap="jet",vmin=0,vmax=0.05)
    ax[1].set_title("Pred")
    # Add a colorbar
    fig.colorbar(cax1, fraction=0.03, pad=0.04,orientation='horizontal', location='top')
    fig.colorbar(cax2, fraction=0.03, pad=0.04,orientation='horizontal', location='top')
    #abs
    cax3=ax[2].imshow(np.abs(ground_truth[0,:,0,:].cpu().numpy()-pred_data[0,:,0,:].cpu().numpy()),
                      cmap="jet",
                      vmin=0,vmax=0.01)
    fig.colorbar(cax3,fraction=0.03, pad=0.04,orientation='horizontal', location='top')
    ax[2].set_title("Abs error")
    #记录global 误差test_data [batch,3,640,300]，640为时间步
    global_metric.Calulate(ground_truth[:,:,0,:].cpu().numpy(),pred_data[:,:,0,:].cpu().numpy())
    #记录local 误差
    local_metric.Calulate(ground_truth[:,:,0,100:200].cpu().numpy(),pred_data[:,:,0,100:200].cpu().numpy())
    


    plt.tight_layout()
    if os.path.exists(folder):
       plt.savefig(f"{folder}/test{epoch}.png")
       plt.close()
    else:
      os.makedirs(folder)
      plt.savefig(f"{folder}/test{epoch}.png")
      plt.close()
      print("create folder")

    print("Mean_step_test_loss",Mean_step_loss,flush=True)
    
    return Mean_step_loss,global_metric,local_metric
    
def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
import numpy as np     

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('需要布尔值。')



if __name__ == "__main__":
    #记录日期
  print("当前工作目录:", os.getcwd())
  # 创建 ArgumentParser 对象
  parser = argparse.ArgumentParser(description="Process some integers.")
  # 添加alpha参数
  parser.add_argument('--accelerator',type = bool,default=False,help="accelerator")
  parser.add_argument('--alpha', type = float, help = 'alpha to control global and local weight ')
  # 添加fno mode参数
  parser.add_argument('--modes', type = int, default = 32, help='modes of fno')
  # 添加seed参数
  parser.add_argument('--seed', type = int, default = 1234,help = 'seeds exprs')
  #保存的参数
  parser.add_argument('--dat', type = str,  required = True,help = 'description of the exprs and save')
  parser.add_argument('--data_folder',type = str,default="None",help="Dataset_path")
  parser.add_argument("--pid",type=str2bool,help = "pid effect(默认关闭)")
  parser.add_argument("--Trunk",type = str, required = True, help="trunk net")

  #记录的epoch
  parser.add_argument('--save_epoch',type = int,default = 1000)
  # 添加可选参数的train
  Train=parser.add_argument_group('Train',description = 'Train paremeters')

  # 向另一个参数组中添加参数，基本都是默认
  Train.add_argument('--epoch',type = int,default = 5000, help='epoch of train')
  Train.add_argument('--batch_size',type = int,default = 500,help='batch_size of train')
  Train.add_argument('--lr',type = float,default = 1e-3, help = 'learning rate of train')
  Load = False   #默认已经整理好:False

  # 解析命令行参数
  args = parser.parse_args()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # 添加位置参数
  if args.pid ==True:
    print("pid=true")

  pid_alpha = PID_alpha(alpha = args.alpha,pid_en=args.pid)
    
  
  fno_modes= args.modes #for wno is level 1/2/3/4
  
  seed = args.seed
  num_epochs= args.epoch  # 训练 epcoh

  set_seed(args.seed)
  dat = args.dat
  
  # 将args对象转换为字典
  args_dict = vars(args)
  model_save_path = f"Model_out/{dat}"
  
  if not os.path.exists(model_save_path):
      os.makedirs(model_save_path)

  # 将字典保存为JSON文件
  with open(f'Model_out/{dat}/args.json', 'w') as json_file:
      json.dump(args_dict, json_file, indent=4)
      
  # copy 此文件到保存的文件夹作为存档
  shutil.copy("Operator/Train_NCHD.py",f"Model_out/{dat}/Train_NCHD.py")
  model_save_path=f"Model_out/{dat}"

  # 每次加载 .mat 文件 和保存,加大随机性
  if Load == False:
    All_mat_file= return_name_list(folder=args.data_folder)
    # 打乱文件列表以确保随机性
    np.random.shuffle(All_mat_file)

    # 计算训练集的大小（70%）
    train_size = int(len(All_mat_file) * 0.7)
    
    # 分割列表为训练集和测试集
    train_files = All_mat_file[:train_size]
    test_files = All_mat_file[train_size:]

    #Test_mat_file= return_name_list(folder="Data/Test")
    train_torch_data,train_Conditions = Read_Mat_4torch(mat_file=train_files)._read_mat()

    test_torch_data,test_Conditions = Read_Mat_4torch(mat_file=test_files)._read_mat()
    
    
    #归一化输入数据，记得test的时候反归一化，但测试感觉loss下降的差不多
    print("before_norm",train_torch_data.shape) #norm torch.Size([b, 3,640, 300]) 

    #***** 维度变化，输入【b,1,1,300】-> 输出[b,640,1,300]
    #train
    train_g_t = train_torch_data[:,2:3,:,:]# ground_truth torch.Size([b, 1,640, 300])
    train_g_t = train_g_t.permute(0,2,1,3) #=[b,640,1,300]
    
    train_input_data = train_torch_data[:,2:3,0:1,:] # ini([b, 1,1, 300]) 0:1 is time
    train_input_data = train_input_data.permute(0,2,1,3) # =[20,1,1,300】时间维度在前面
    
    #gaussion Normalizer
    
    gt_normalizer = utilities3.UnitGaussianNormalizer(train_g_t)
    
    train_g_t = gt_normalizer.encode(train_g_t)
    
    input_normalizer = utilities3.UnitGaussianNormalizer(train_input_data)
    train_input_data = input_normalizer.encode(train_input_data)
    
    train_OE_Dataset = OE_Dataset(input_data = train_input_data,
                                  input_cond = train_Conditions,
                                  out = train_g_t)
    # test 
    test_g_t = test_torch_data[:,2:3,:,:] # ground_truth torch.Size([b, 1,640, 300])
    test_g_t = test_g_t.permute(0,2,1,3) #[b,640,1,300]
    test_input_data = test_torch_data[:,2:3,0:1,:] # input_data torch.Size([b, 1,1, 300]) 0:1 is time
    test_input_data = test_input_data.permute(0,2,1,3) #变成[20,1,1,300】时间维度在前面
    
    test_g_t = gt_normalizer.encode(test_g_t) #[batch,640,1,300]
    test_input_data = input_normalizer.encode(test_input_data)
    

    test_OE_Dataset = OE_Dataset(input_data = test_input_data,
                                 input_cond = test_Conditions,
                                 out = test_g_t)
    
    #保存
    torch.save(train_OE_Dataset, 'Data/NCHD_pt/train_OE_Dataset.pt')
    torch.save(test_OE_Dataset, 'Data/NCHD_pt/test_OE_Dataset.pt')
  

  #定义模型和相关优化器
  train_loader = DataLoader(train_OE_Dataset,batch_size = args.batch_size,shuffle=True)
  test_loader = DataLoader(test_OE_Dataset,batch_size=args.batch_size,shuffle=True)
  mNO = MFNO(Trunk=args.Trunk)
  mse=torch.nn.MSELoss()
  optimzer = torch.optim.Adam(mNO.parameters(),lr=args.lr)
  # 创建学习率调度器
  scheduler = torch.optim.lr_scheduler.StepLR(optimzer, step_size=100, gamma=0.99)

  alpha_list =[]
  losses = [] # 提前分配内存空间
  epoch_list = []

  test_losses = []
   
  print("start train-----")

  # 加速
  if (args.accelerator == True):
    
    print("accelerator")
    from accelerate import Accelerator
    train_loader, test_loader, mNO,optimzer = accelerator.prepare(train_loader, test_loader, mNO, optimzer)
    
  else:
    
    mNO.to(device)
    gt_normalizer.cuda()
    input_normalizer.cuda()




  for epoch in range(num_epochs):
    
    print("epoch:",epoch,flush=True)

    with tqdm(total= len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:

      for i, (ini_data, condition,ground_truth) in enumerate(train_loader):

        loss = 0.0
        ini_data = ini_data #[batch,1,1,300] first1 is time
        condition = condition#[batch,2(A and L)]
        ground_truth = ground_truth
        

        final_out = mNO(ini_data,condition) #should:[20,640,3,300]
        

       
        #第2个的维度表示eta 
        outside_indices_loss = mse(final_out[:, :, :, 0:100], ground_truth[:,:,:,0:100]) + \
                              mse(final_out[:, :, :, 200:300], ground_truth[:,:,:,200:300])
                          
        # 计算 100:200 区间内的损失贡献 slope
        slope_loss = mse(final_out[:, :, :, 100:200], ground_truth[:,:,:,100:200])

        
      
        loss = (1-pid_alpha.alpha)*outside_indices_loss +  pid_alpha.alpha * slope_loss
    
        optimzer.zero_grad()
        
        if args.accelerator == True:
          accelerator.backward(loss)
        else:
          loss.backward()
          
        #call 方法
        pid_alpha(outside_indices_loss.item()) # 因为epect=0
          
        optimzer.step()

        pbar.update(1)
      
      print(f"epoch:{epoch},train_loss:{loss}",flush=True)
      
      losses.append(loss.item())
      epoch_list.append(epoch)
      alpha_list.append(pid_alpha.alpha)
      # 在每个epoch结束后更新学习率
      scheduler.step()

      if epoch % args.save_epoch == 0 :
        #训练loss

        fig,ax = plt.subplots(1,2,figsize=(10,6))
        if args.pid:
          ax[0].plot(epoch_list, losses, label= f"$K_p={pid_alpha.K_p}$,$K_i={pid_alpha.K_i}$,$K_d={pid_alpha.K_d}$")
        else:
          ax[0].plot(epoch_list,losses,label = f"Loss" )
          
        ax[0].set_yscale('log')
        ax[0].legend(loc= "upper left",fontsize=12)
        ax[0].set_title("train loss with epochs")
        
        ax[1].plot(epoch_list,alpha_list,label = r'$\alpha$')
        ax[1].legend(loc= "upper left",fontsize=12)
        ax[0].set_xlabel(f'x1 epoch ')  # 添加x轴的标签
        ax[1].set_xlabel(f'x1 epoch ')  # 添加x轴的标签
        
        ax[1].set_title(r"$\alpha$ with epochs")
        plt.tight_layout()
        plt.savefig(f"{model_save_path}/PID{args.pid}_effect.png")
        
        plt.close()
        
        
  
        test_mean_step_loss,Global_metric,Local_metric = test(
                                                            folder=model_save_path,
                                                            epoch=epoch,
                                                            test_loader=test_loader,
                                                            device=device,
                                                           )
        
                                                    
        test_losses.append( test_mean_step_loss )
        if epoch == 0:
          prev_loss= losses[-1]
        else:
          prev_loss= losses[-2]
          
        if prev_loss > 1.1 * loss.item() :
          print("save best model",flush=True)
          torch.save(mNO.state_dict(), f"{model_save_path}/mno_ckpt.pth")
          
        if epoch >0:
          
          plot_test_performance(model_save_path,epoch,
                                Local_metric=Local_metric,Global_metric=Global_metric,title="Global and Local",
                                test_losses=test_losses)

      # 保存训练损失和测试损失 npz
      if epoch % args.save_epoch == 0 and epoch != 0:
        
        print("save",flush=True)
        np.savez(f"{model_save_path}/seed_{seed}_modes{fno_modes}_alpha_{args.alpha}_pid_{args.pid}_losses_with_epoch.npz",
                  training_loss=losses, test_loss=test_losses,global_metric=Global_metric.Global_dict,
                  local_metric=Local_metric.Local_dict,epoch=epoch,alpha=alpha_list)
        
  print("done")

        
          
    

   