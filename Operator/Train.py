
from neuralop.models import FNO
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
# from Evaluate import calculate_psnr,calculate_ssim
class branch_net(nn.Module):
    def __init__(self,input,hidden,output):
        super(branch_net,self).__init__()
            
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
    
    num_points= int (epoch / 10) + 1
      
    fig,ax= plt.subplots(4,1,figsize=(12,12))
    
    ax[0].plot(range(num_points),g_PSNR, label="Global PSNR", linestyle='-',
               linewidth=3,marker="o")
    ax[0].plot(range(num_points),l_PSNR, label="Local PSNR", linestyle='--',
               linewidth=3,marker="s")
    # 设置刻度标签，确保标签反映实际的epoch数
    ax[0].set_title(f"PSNR_{epoch}_"+kwargs["title"]+f"_alpha_{alpha}")
    ax[0].legend()
    ax[0].set_xlabel('x10 epoch ')  # 添加x轴的标签
    
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
def test(folder,epoch,test_loader,device,norm_transform):

    with torch.no_grad():
      mse=torch.nn.MSELoss()
      loss=0
      for i,(data,condition) in enumerate(test_loader):
        
        pred_data = np.zeros_like(data.cpu().numpy())
        pred_data=torch.from_numpy(pred_data)
        condition=condition.to(device)
        
        data=data.to(device)
        test_data=data
        t_steps= data.shape[2]-1
        #对整个序列做变换
        fno_out=fno(data)#[batch, 3, 640, 300]，640是时间步
      
        out=bran_nn(condition)
        out=out.unsqueeze(-1)
        out=out.unsqueeze(-1)
  
        final_out=fno_out*out
      
        # #反归一化
        # if norm_transform is not None:
        #   final_out=norm_transform.inverse(final_out)
        pred_data = final_out
        
        loss+=mse(final_out,data)
          
    Mean_step_loss=(loss/t_steps).item()
    fig,ax= plt.subplots(1,3,figsize=(12,12))
    #画图预测第一个batch的数据
    cax1=ax[0].imshow(test_data[0,2,:,:].cpu().numpy(),cmap="jet",vmin=0,vmax=0.6)
    ax[0].set_title("True"+f"A_{condition[0,0].item():.2f}"+f"L_{condition[0,1].item():.2f}") 
    cax2=ax[1].imshow(pred_data[0,2,:,:].cpu().numpy(),cmap="jet",vmin=0,vmax=0.6)
    ax[1].set_title("Pred")
    # Add a colorbar
    fig.colorbar(cax1, fraction=0.046, pad=0.04,orientation='horizontal', location='top')
    fig.colorbar(cax2, fraction=0.046, pad=0.04,orientation='horizontal', location='top')
    #abs
    cax3=ax[2].imshow(np.abs(test_data[0,2,:,:].cpu().numpy()-pred_data[0,2,:,:].cpu().numpy()),
                      cmap="jet",
                      vmin=0,vmax=0.1)
    fig.colorbar(cax3,fraction=0.046, pad=0.04,orientation='horizontal', location='top')
    ax[2].set_title("Abs error")
    #记录global 误差test_data [batch,3,640,300]，640为时间步
  

    global_metric.Calulate(test_data[0,2,:,:].cpu().numpy(),pred_data[0,2,:,:].cpu().numpy())
    #记录local 误差
    local_metric.Calulate(test_data[0,2,:,100:200].cpu().numpy(),pred_data[0,2,:,100:200].cpu().numpy())
  
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

if __name__ == "__main__":
    #记录日期
  print("当前工作目录:", os.getcwd())
  # 创建 ArgumentParser 对象
  parser = argparse.ArgumentParser(description="Process some integers.")
  # 添加alpha参数
  parser.add_argument('--alpha', type=float, help='alpha to control global and local weight ')
  # 添加fno mode参数
  parser.add_argument('--modes', type=int, default=32,help='modes of fno')
  # 添加seed参数
  parser.add_argument('--seed', type=int, default=1234,help='seeds exprs')
  #保存的参数
  parser.add_argument('--dat', type=str,  required=True,help='description of the exprs and save')
  # 添加可选参数的train
  Train=parser.add_argument_group('Train','Train paremeters')

  # 向另一个参数组中添加参数，基本都是默认
  Train.add_argument('--epoch',type=int,default=5000, help='epoch of train')
  Train.add_argument('--batch_size',type=int,default=20, help='batch_size of train')
  Train.add_argument('--lr',type=float,default=0.001, help='learning rate of train')
  Train.add_argument("--data_folder",type=str,default="Data/Train/Train1d",help="train folder")
  # 解析命令行参数
  args = parser.parse_args()
  
  # 添加位置参数
  alpha= args.alpha
  fno_modes= args.modes
  seed = args.seed
  num_epochs= args.epoch  # 训练 epcoh
  
  print(num_epochs)

  set_seed(args.seed)
  dat=args.dat
  # 将args对象转换为字典
  args_dict = vars(args)
  model_save_path=f"Model_out/{dat}"
  
  if not os.path.exists(model_save_path):
      os.makedirs(model_save_path)

  # 将字典保存为JSON文件
  with open(f'Model_out/{dat}/args.json', 'w') as json_file:
      json.dump(args_dict, json_file, indent=4)
      
  # copy 此文件到保存的文件夹作为存档
  shutil.copy("Operator/Train.py",f"Model_out/{dat}/Train.py")
  model_save_path=f"Model_out/{dat}"
  device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # 加载 .mat 文件
  All_mat_file= return_name_list(folder=args.data_folder)
  # 打乱文件列表以确保随机性
  np.random.shuffle(All_mat_file)

  # 计算训练集的大小（70%）
  train_size = int(len(All_mat_file) * 0.7)
  
  # 分割列表为训练集和测试集
  train_files = All_mat_file[:train_size]
  test_files = All_mat_file[train_size:]

  #Test_mat_file= return_name_list(folder="Data/Test")
  train_np_data,train_OE_Data,norm_transform=Read_Mat_4torch(mat_file=train_files)._read_mat()


  test_np_data,test_OE_Data,norm_transform=Read_Mat_4torch(mat_file=test_files)._read_mat()
  
  train_loader=DataLoader(train_OE_Data,batch_size=args.batch_size,shuffle=True)
  test_loader=DataLoader(test_OE_Data,batch_size=args.batch_size,shuffle=True)
  
  #3个channnel 是三个形态
  fno=FNO(n_modes=(fno_modes,fno_modes),hidden_channels=12,in_channels=3,out_channels=3).to(device)
  bran_nn=branch_net(2,50,1).to(device)
  mse=torch.nn.MSELoss()
  optimzer1=torch.optim.Adam(fno.parameters(),lr=args.lr)
  optimzer2=torch.optim.Adam(bran_nn.parameters(),lr=args.lr)

  losses = [] # 提前分配内存空间

  test_losses = []
   
  print("start train")
  for epoch in range(num_epochs):
    
    print("epoch:",epoch,flush=True)
    
    for i,(data,condition) in enumerate(train_loader):
      loss=0
      data=data.to(device) #[batch,3,t_steps,300]
      condition=condition.to(device) #[batch,2(A and L)]
      t_steps=data.shape[2]-1

      #对序列进行fno
      fno_out=fno(data) #[4, 3, 640, 300]，640是时间步

      expand_size=data.shape[-2] #300 =100*3
      out=bran_nn(condition)
      out=out.unsqueeze(-1)
      out=out.unsqueeze(-1)
      #([batch, 3, 1, 300])
      final_out=fno_out*out

      #重点关注alpha对于 shelf
      # shelf 在100:200
      # 计算 100:200 区间以外的损失贡献
      
      outside_indices_loss = mse(final_out[:, :, :, :100], data[:, :, :, :100]) + \
                            mse(final_out[:, :, :, 200:], data[:, :, :, 200:])
      # 计算 100:200 区间内的损失贡献
      inside_indices_loss = mse(final_out[:, 2, :, 100:200], data[:, 2, :, 100:200])
                            
      loss+=(1-alpha)*outside_indices_loss+ \
            alpha * mse(final_out[:,2,:,100:200],data[:,2,:,100:200])

      optimzer1.zero_grad()
      optimzer2.zero_grad()

      loss.backward()
      optimzer1.step()
      optimzer2.step()

    
    print(f"epoch:{epoch},loss:{loss}",flush=True)
    

    if epoch % 10 == 0 :
      #训练loss
      losses.append(loss.item())
      
      test_mean_step_loss,Global_metric,Local_metric=test(
                                                          folder=model_save_path,
                                                          epoch=epoch,
                                                          test_loader=test_loader,
                                                          device=device,
                                                          norm_transform=None)
                                                  
      test_losses.append( test_mean_step_loss )
      if epoch == 0:
        prev_loss= losses[-1]
      else:
        prev_loss= losses[-2]
      
      if prev_loss > loss.item() :
        print("save best model")
        torch.save(fno.state_dict(), f"{model_save_path}/fno.pth")
        torch.save(bran_nn.state_dict(), f"{model_save_path}/branch_nn.pth")
        
      if epoch >0:
        
        plot_test_performance(model_save_path,epoch,
                              Local_metric=Local_metric,Global_metric=Global_metric,title="Global and Local",
                              training_losses=losses,test_losses=test_losses)


    # 保存训练损失和测试损失 npz
    if epoch % 10 == 0 and epoch != 0:
      
      print("save")
      np.savez(f"{model_save_path}/seed_{seed}_modes{fno_modes}_alpha_{alpha}_losses_with_epoch.npz",
                training_loss=losses, test_loss=test_losses,global_metric=Global_metric.Global_dict,
                local_metric=Local_metric.Local_dict,epoch=epoch)
      
  print("done")

      
         
   

   