
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
        self.kd = kd if kd is not None else 0.1
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
      if Trunk == "FNO":
        self.trunk_nn = FNO(n_modes=(fno_modes,fno_modes),hidden_channels=12,in_channels=1,out_channels=t_steps)
        
      elif Trunk == "UNO":
        self.trunk_nn = UNO(model = UNO(in_channels=1,out_channels=t_steps, hidden_channels=12, projection_channels=12,uno_out_channels = [32,64,64,64,32], \
            uno_n_modes= [[fno_modes,fno_modes],[fno_modes//2,fno_modes//2],[fno_modes//2,fno_modes//2],[fno_modes//2,fno_modes//2],[fno_modes,fno_modes]], uno_scalings=  [[1.0,1.0],[0.5,0.5],[1,1],[2,2],[1,1]],\
            horizontal_skips_map = None, n_layers = 5, domain_padding = 0.2))
        
      elif Trunk == "WNO":
        assert "This assertion is not implemented" 
      
      self.condition_norm = nn.BatchNorm1d(infeature)  # 归一化层
      self.ini_norm = nn.BatchNorm2d(1)

  def forward(self,x,condition):
      #对序列进行fno
      #[batch,1,3,300]
      x_norm =self.ini_norm(x)

      condition_normal= self.condition_norm(condition)
      #norm
      trunk_out = self.trunk_nn(x_norm) #[4, 3, 640, 300]，640是时间步
      
      out = self.bran_nn(condition_normal)
  
      out = out.unsqueeze(-1)
      out = out.unsqueeze(-1)
      #([batch, 1, 640, 300])
      final_out = trunk_out*out
      #[batch, 1, 640, 300]
      final_out = final_out.permute(0,2,1,3)
   
      return final_out
  def stat_trunk_params(self):
       # 遍历模型参数
      self.total_params=0
      for param in self.trunk_nn.parameters():
          self.total_params += param.numel()
      return self.total_params
      
    

def Get_4_folder(train_np_data,test_np_data,name="NCHD"):
   # Define your folder name
  folder_name = f'Data/{name}'+ '_pt'
   # Check if the folder exists, and if not, create it
  if not os.path.exists(folder_name):
      os.makedirs(folder_name)
      # Define your folder and file name
  file_name = 'Train_dataset.pt'
  file_path = os.path.join(folder_name, file_name)

  # Check if the specific file exists in the folder
  if os.path.isfile(file_path):
      print(f"The file {file_name} exists in {folder_name}.")
      train_data = torch.load(f'{folder_name}/Train_dataset.pt')
      train_dataset = train_data['train_dataset']

      # Load the test dataset
      test_data = torch.load(f'{folder_name}/Test_dataset.pt')
      test_dataset = test_data['test_dataset']
  else:
      print(f"The file {file_name} does not exist in {folder_name}.")
      # Save the train dataset and normalization transform
      torch.save({
          'train_dataset': train_np_data,
      }, f'{folder_name}/Train_dataset.pt')

      # Save the test dataset
      torch.save({
          'test_dataset': test_np_data
      }, f'{folder_name}/Test_dataset.pt')

  
  return train_dataset,test_dataset
       
def plot_test_performance(folder,epoch,**kwargs):
    #SSIM是基于对亮度、对比度和结构三个不同维度的比较计算得出的。
    # 具体来说，SSIM越接近1，意味着两幅图像在视觉上越相似；
    # SSIM越接近0，意味着两幅图像在视觉上的差异越大。
    
    g_SSIM = kwargs["Global_metric"].Global_dict["ssim"]
    g_PSNR = kwargs["Global_metric"].Global_dict["psnr"]
    g_MSE = kwargs["Global_metric"].Global_dict["mse"]
    
    #local
    l_SSIM = kwargs["Local_metric"].Local_dict["ssim"]
    l_PSNR = kwargs["Local_metric"].Local_dict["psnr"]
    l_MSE = kwargs["Local_metric"].Local_dict["mse"]
    
    #loss
    training_losses = kwargs["training_losses"]
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
    
    ax[2].plot(range(num_points),g_MSE, label="Global MSE",linestyle='-',
               linewidth=3,marker="o")
    ax[2].plot(range(num_points),l_MSE, label="Local MSE",linestyle='--',
               linewidth=3,marker="s")
    ax[2].set_yscale('log')
    ax[2].set_title(f"MSE_{epoch}_"+kwargs["title"])
    ax[2].legend()
    
    ax[3].plot(range(num_points),training_losses, label="Training loss", 
             linewidth=3,linestyle='-')
    ax[3].plot(range(num_points),test_losses, label="Test Mean step loss",linewidth=3,
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
      for i,(data,condition) in enumerate(test_loader):
        
        # pred_data = np.zeros_like(data.cpu().numpy())
        # pred_data=torch.from_numpy(pred_data)
        condition=condition.to(device)
        
        data=data.to(device)
        test_data=data
        t_steps= data.shape[2]-1
        #对整个序列做变换
        ini_data = data[:,2:3,0:1,:] #【20,1,1,300】 #2表示eta的维度
        ini_data =ini_data .permute(0,2,1,3)#[20,1,3,300】
        
  
        
        final_out= mFNO (ini_data,condition) #[20,1,640,300]
        
        final_out = x_normalizer.decode(final_out)
        data= x_normalizer.decode(data)
        
        pred_data =  final_out
        
        loss = mse (final_out[:,:,:,:],data[:,2:3,:,:])


    Mean_step_loss= loss.item()
    fig,ax= plt.subplots(1,3,figsize=(12,8))
    #画图预测第一个batch的数据tst_data[0,2,:,:],2表示eta
    cax1=ax[0].imshow(test_data[0,2,:,:].cpu().numpy(),cmap="jet",vmin=0,vmax=0.05)
  
    ax[0].set_title("True"+f"A_{condition[0,0].item():.2f}"+f"_L_{condition[0,1].item():.2f}") 
    cax2=ax[1].imshow(pred_data[0,0,:,:].cpu().numpy(),cmap="jet",vmin=0,vmax=0.05)
    ax[1].set_title("Pred")
    # Add a colorbar
    fig.colorbar(cax1, fraction=0.03, pad=0.04,orientation='horizontal', location='top')
    fig.colorbar(cax2, fraction=0.03, pad=0.04,orientation='horizontal', location='top')
    #abs
    cax3=ax[2].imshow(np.abs(test_data[0,2,:,:].cpu().numpy()-pred_data[0,0,:,:].cpu().numpy()),
                      cmap="jet",
                      vmin=0,vmax=0.01)
    fig.colorbar(cax3,fraction=0.03, pad=0.04,orientation='horizontal', location='top')
    ax[2].set_title("Abs error")
    #记录global 误差test_data [batch,3,640,300]，640为时间步
    global_metric.Calulate(test_data[:,2,:,:].cpu().numpy(),pred_data[:,0,:,:].cpu().numpy())
    #记录local 误差
    local_metric.Calulate(test_data[:,2,:,100:200].cpu().numpy(),pred_data[:,0,:,100:200].cpu().numpy())
    


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
  parser.add_argument('--alpha', type=float, help='alpha to control global and local weight ')
  # 添加fno mode参数
  parser.add_argument('--modes', type=int, default=32,help='modes of fno')
  # 添加seed参数
  parser.add_argument('--seed', type=int, default=1234,help='seeds exprs')
  #保存的参数
  parser.add_argument('--dat', type=str,  required=True,help='description of the exprs and save')
  parser.add_argument('--data_folder',type=str,default="None",help="Dataset_path")
  parser.add_argument("--pid",type=str2bool,help="pid effect(默认关闭)")

  #记录的epoch
  parser.add_argument('--save_epoch',type=int,default=2)
  # 添加可选参数的train
  Train=parser.add_argument_group('Train',description='Train paremeters')

  # 向另一个参数组中添加参数，基本都是默认
  Train.add_argument('--epoch',type=int,default=5000, help='epoch of train')
  Train.add_argument('--batch_size',type=int,default=50, help='batch_size of train')
  Train.add_argument('--lr',type=float,default=1e-3, help='learning rate of train')
  Load = False   #默认已经整理好:False

  # 解析命令行参数
  args = parser.parse_args()
  

  # 添加位置参数
  if args.pid ==True:
    print("pid=true")

  pid_alpha = PID_alpha(alpha = args.alpha,pid_en=args.pid)
    
  
  fno_modes= args.modes
  
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

  # 加载 .mat 文件
  if (Load == False):
    All_mat_file= return_name_list(folder=args.data_folder)
    # 打乱文件列表以确保随机性
    np.random.shuffle(All_mat_file)

    # 计算训练集的大小（70%）
    train_size = int(len(All_mat_file) * 0.7)
    
    # 分割列表为训练集和测试集
    train_files = All_mat_file[:train_size]
    test_files = All_mat_file[train_size:]

    #Test_mat_file= return_name_list(folder="Data/Test")
    train_np_data,train_OE_Data,norm_transform = Read_Mat_4torch(mat_file=train_files)._read_mat()


    test_np_data,test_OE_Data,norm_transform = Read_Mat_4torch(mat_file=test_files)._read_mat()

  
  train_OE_Data = None
  test_OE_Data = None
  # 读取 .pt
  train_OE_Data,test_OE_Data = Get_4_folder(train_OE_Data,test_OE_Data)
  

  #归一化
  x_normalizer = utilities3.UnitGaussianNormalizer(train_OE_Data)
  train_OE_Data = x_normalizer.encode(train_OE_Data)
  test_OE_Data = x_normalizer.encode(test_OE_Data) # 归一化到dataloader

  
  train_loader = DataLoader(train_OE_Data,batch_size = args.batch_size,shuffle=True)
  test_loader = DataLoader(test_OE_Data,batch_size=args.batch_size,shuffle=True)
  
  #3个channnel 是三个形态

  mFNO = MFNO()
  mse=torch.nn.MSELoss()
  optimzer=torch.optim.Adam(mFNO.parameters(),lr=args.lr)

  alpha_list =[]
  losses = [] # 提前分配内存空间
  epoch_list = []

  test_losses = []
   
  print("start train-----")

  # 加速
  if (args.accelerator == True):
    print("accelerator")
    from accelerate import Accelerator
    train_loader, test_loader, mfno,optimzer = accelerator.prepare(train_loader, test_loader, mFNO, optimzer)
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mFNO.to(device)



  for epoch in range(num_epochs):
    
    print("epoch:",epoch,flush=True)

    with tqdm(total= len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:

      for i, (ini_data, condition,ground_truth) in enumerate(train_loader):

        loss = 0.0
        ini_data = ini_data.to(device) #[batch,3,1,300] 1 is t0

        condition=condition.to(device) #[batch,2(A and L)]
      
        # ini_data = data[:,2:3,0:1,:] #【20,1,1,300】 #2:3表示eta的维度,0:1 表示时间的维度
        ini_data =ini_data .permute(0,2,1,3)#[20,1,3,300】#变化后【b,t,coords,3parts】
        
        final_out = mFNO(ini_data,condition) #[20,1,640,300]
        
        data = x_normalizer.decode(data)
        final_out = x_normalizer.decode(final_out) #  反归一化
        print("final_out",final_out.shape) #final_out torch.Size([50, 1, 640, 300])
       
        #第2个的维度表示eta 
        outside_indices_loss = mse(final_out[:, :, :, 0:100], ground_truth) + \
                              mse(final_out[:, :, :, 200:300], ground_truth)
                          
        # 计算 100:200 区间内的损失贡献 slope
        slope_loss = mse(final_out[:, :, :, 100:200], ground_truth)
        
      
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

      if epoch % args.save_epoch == 0 :
        #训练loss

        fig,ax = plt.subplots(1,2,figsize=(10,6))
        if args.pid:
          ax[0].plot(epoch_list, losses, label=f"PID $K_p={pid_alpha.K_p}$,$K_i={pid_alpha.K_i}$,$K_d={pid_alpha.K_d}$")
        else:
          ax[0].plot(epoch_list,losses,label = f"Loss" )
          
          
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
          torch.save(mFNO.state_dict(), f"{model_save_path}/mfno.pth")
          
        if epoch >0:
          
          plot_test_performance(model_save_path,epoch,
                                Local_metric=Local_metric,Global_metric=Global_metric,title="Global and Local",
                                training_losses=losses,test_losses=test_losses)

      # 保存训练损失和测试损失 npz
      if epoch % args.save_epoch == 0 and epoch != 0:
        
        print("save",flush=True)
        np.savez(f"{model_save_path}/seed_{seed}_modes{fno_modes}_alpha_{args.alpha}_pid_{args.pid}_losses_with_epoch.npz",
                  training_loss=losses, test_loss=test_losses,global_metric=Global_metric.Global_dict,
                  local_metric=Local_metric.Local_dict,epoch=epoch,alpha=alpha_list)
        
  print("done")

        
          
    

   