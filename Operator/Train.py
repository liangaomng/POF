
from neuralop.models import FNO
from torch import nn
from Dataset_torch import Read_Mat_4torch
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from Evaluate import calculate_psnr,calculate_ssim
class branch_net(nn.Module):
    def __init__(self,input,hidden,output):
        super(branch_net,self).__init__()
            
        self._net = nn.Sequential(
                     nn.Linear(input, hidden),
                     nn.Tanh(),
                     nn.Linear(hidden, hidden),
                     nn.Tanh(),
                     nn.Linear(hidden, output),
                     nn.PReLU(),
                    )
    def forward(self,x):
       
        out=self._net(x)
        return out
def plot_losses(training_losses, test_losses, folder, epoch):
    plt.figure(figsize=(8, 6))
    plt.plot(range(epoch),training_losses[:epoch], label="Training loss", linewidth=3)
    plt.plot(range(epoch),test_losses[:epoch], label="Test loss", linewidth=3)
    plt.legend()
    plt.yscale('log')
    plt.title(f"Losses up to Epoch")
    plt.savefig(f"{folder}/losses_up_to_epoch.png")
    plt.close()
def plot_test_performance(SSIM:list,PSNR,MSE,folder,epoch):
    #SSIM是基于对亮度、对比度和结构三个不同维度的比较计算得出的。
    # 具体来说，SSIM越接近1，意味着两幅图像在视觉上越相似；
    # SSIM越接近0，意味着两幅图像在视觉上的差异越大。
    fig,ax= plt.subplots(3,1,figsize=(12,12))
    ax[0].plot(range(epoch),PSNR[:epoch], label="Test PSNR", linestyle='--',
               linewidth=3,marker="o")
    ax[0].set_title("PSNR epoch")
    ax[0].legend()
    ax[1].plot(range(epoch),SSIM[:epoch],label="Test SSIM",linestyle='--',
               linewidth=3,marker="s")
    ax[1].set_title("SSIM epoch")
    ax[1].legend()
    ax[2].plot(range(epoch),MSE[:epoch], label="Test MSE",linestyle='--',
               linewidth=3,marker="*")
    ax[2].set_title("MSE epoch")
    ax[2].legend()
    plt.title(f"Losses up to Epoch")
    plt.savefig(f"{folder}/Test_performance.png")
    plt.close()
   
# 测试 and plot 
def test(folder,epoch,test_data,condition,device):

    test_data=torch.from_numpy(test_data)

    pred_data=np.zeros_like(test_data.cpu().numpy())
    test_data=test_data.to(device)
    condition=condition.to(device)

    with torch.no_grad():
      mse=torch.nn.MSELoss()
      loss=0
      batch,t_steps,_,value=test_data.shape
      
      for j in range(t_steps-1):

         fno_out=fno(test_data[:,j:j+2,:,:])
      
   
         expand_size=data.shape[-2] #300 =100*3
         out=bran_nn(condition)

         out=out.unsqueeze(-2)
         out=out.repeat(1,1,expand_size,1)
         
         #([2, 2, 300, 3])
         final_out=fno_out*out
      
         pred_data[:,j:j+2,:,:]=final_out.cpu().numpy()

         loss+=mse(final_out,test_data[:,j:j+2,:,:])

    Mean_step_loss=(loss/t_steps).item()
    fig,ax= plt.subplots(1,3,figsize=(12,12))
    #预测第一个batch的数据
    cax1=ax[0].imshow(test_data[0,:,:,2].cpu().numpy(),cmap="jet",vmin=0,vmax=0.6)
    ax[0].set_title("True")
    cax2=ax[1].imshow(pred_data[0,:,:,2],cmap="jet",vmin=0,vmax=0.6)
    ax[1].set_title("Pred")
    #colorbar 帮我写一个      # Add a colorbar
    fig.colorbar(cax1, fraction=0.046, pad=0.04,orientation='horizontal', location='top',)
    fig.colorbar(cax2, fraction=0.046, pad=0.04,orientation='horizontal', location='top')
    #abs
    cax3=ax[2].imshow(np.abs(test_data[0,:,:,2].cpu().numpy()-pred_data[0,:,:,2]),
                      cmap="jet",
                      vmin=0,vmax=0.1)
    fig.colorbar(cax3,fraction=0.046, pad=0.04,orientation='horizontal', location='top')
    ax[2].set_title("Abs error")
    SSIM=calculate_ssim(test_data[0,:,:,2].cpu().numpy(),pred_data[0,:,:,2])
    PSNR,MSE=calculate_psnr(test_data[0,:,:,2].cpu().numpy(),pred_data[0,:,:,2])
    if os.path.exists(folder):
       
   
       plt.savefig(f"{folder}/test{epoch}.png")
       plt.close()
    else:
      os.makedirs(folder)
      plt.savefig(f"{folder}/test{epoch}.png")
      plt.close()
      print("create folder")
    print("test_loss",Mean_step_loss)
    return Mean_step_loss,SSIM,PSNR,MSE
    
def set_seed(seed):
      torch.manual_seed(seed)
      if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)

      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
import numpy as np     

if __name__ == "__main__":
    #记录日期
   print("当前工作目录:", os.getcwd())
   alpha=0.1

   dat="24_1_30"
   model_save_path=f"Model_out/{dat}"
   device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   # 加载 .mat 文件
   Train_mat_file = ['/liujinxin/lam/OE/Data/A_0.3_L_2.0_wave_data.mat',
                     '/liujinxin/lam/OE/Data/A_0.3_L_5.0_wave_data.mat']
   
   Test_mat_file=['/liujinxin/lam/OE/Data/A_0.3_L_2.0_wave_data.mat',
                  '/liujinxin/lam/OE/Data/A_0.3_L_5.0_wave_data.mat']
   set_seed(42)
   train_np_data,train_OE_Data=Read_Mat_4torch(mat_file=Train_mat_file)._read_mat()
   test_np_data,test_OE_Data=Read_Mat_4torch(mat_file=Test_mat_file)._read_mat()
   train_loader=DataLoader(train_OE_Data,batch_size=100,shuffle=True,drop_last=True)
   
   #两个网络
   fno=FNO(n_modes=(128,128),hidden_channels=12,in_channels=2,out_channels=2).to(device)
   bran_nn=branch_net(2,50,1).to(device)
   mse=torch.nn.MSELoss()
   optimzer1=torch.optim.Adam(fno.parameters(),lr=0.001)
   optimzer2=torch.optim.Adam(bran_nn.parameters(),lr=0.001)

   num_epochs=1000

   losses = np.zeros(num_epochs)  # 提前分配内存空间
   SS_IM=np.zeros(num_epochs)
   PS_NR=np.zeros(num_epochs)
   M_SE=np.zeros(num_epochs)
   test_losses=np.zeros(num_epochs)
   print("start train")
   for epoch in range(num_epochs):
      print("epoch:",epoch)
      for i,(data,next_t,condition) in enumerate(train_loader):
         loss=0
         data=data.to(device) #[batch,t_steps,300,3]
         condition=condition.to(device)
         next_t=next_t.to(device)
         t_steps=data.shape[1]
         
         for j in range(t_steps):
         
            fno_out=fno(data)
            
            expand_size=data.shape[-2] #300 =100*3
            out=bran_nn(condition)

            out=out.unsqueeze(-2)
            out=out.repeat(1,1,expand_size,1)
            
            #([2, 2, 300, 3])
            final_out=fno_out*out
            print(final_out.shape)
            #重点关注alpha
            loss+=(1-alpha)*mse(final_out,next_t) +alpha *mse(final_out[:,:,100:200,:],next_t[:,:,100:200,:])
      
         optimzer1.zero_grad()
         optimzer2.zero_grad()

         loss.backward()
         optimzer1.step()
         optimzer2.step()


         
      losses[epoch] = loss.item()
   
      if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

      print(f"epoch:{epoch},loss:{loss}")
      if epoch % 1 == 0:
         
        

         test_loss,SSIM,PSNR,MSE=test(
                        folder=model_save_path,
                        epoch=epoch,
                        test_data=test_np_data,
                        condition=test_OE_Data.conditions_to_tensor(),
                        device=device)
         test_losses[epoch]= test_loss
         SS_IM[epoch]=SSIM
         PS_NR[epoch]=PSNR
         M_SE[epoch]=MSE
         
      if epoch % 10 == 0 and epoch != 0:
         torch.save(fno.state_dict(), f"{model_save_path}/fno.pth")
         torch.save(bran_nn.state_dict(), f"{model_save_path}/bran_nn.pth")
         plot_losses(losses, test_losses, model_save_path, epoch)
         plot_test_performance(SS_IM,PS_NR,M_SE,model_save_path,epoch)
      
      
# 保存训练损失和测试损失 npz
np.savez(f"{model_save_path}/losses.npz",
      training_loss=losses, test_loss=test_losses,SSIM=SS_IM,PSNR=PS_NR,MSE=M_SE)
print("done")

      
         
   

   