
from neuralop.models import FNO
from torch import nn
from Dataset_torch import Read_Mat_4torch
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
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


# 测试 and plot 
def test(folder,epoch,test_data,condition):

    test_data=torch.from_numpy(test_data)
    
    pred_data=np.zeros_like(test_data)

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
         
            pred_data[:,j:j+2,:,:]=final_out

            loss+=mse(final_out,test_data[:,j:j+2,:,:])

    Mean_step_loss=(loss/t_steps).item()
    fig,ax= plt.subplots(1,3,figsize=(12,12))
    #预测第一个batch的数据
    ax[0].imshow(test_data[0,:,:,2],cmap="jet")
   
    plt.show()
    if os.path.exists(folder):
       plt.savefig(f"{folder}/test{epoch}.png")
       plt.close()
    else:
      os.makedirs(folder)
      plt.savefig(f"{folder}/test{epoch}.png")
      plt.close()
    print("test_loss",Mean_step_loss)
    return Mean_step_loss
    
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
    dat="24_1_29"
    model_save_path=f"Model/{dat}"
    # 加载 .mat 文件
    Train_mat_file = ['Data/A_0.3_L_2.0_wave_data.mat',
                      'Data/A_0.3_L_5.0_wave_data.mat']
    
    Test_mat_file=['Data/A_0.3_L_5.0_wave_data.mat',
                   'Data/A_0.3_L_5.0_wave_data.mat']
    set_seed(42)
    train_np_data,train_OE_Data=Read_Mat_4torch(mat_file=Train_mat_file)._read_mat()
    test_np_data,test_OE_Data=Read_Mat_4torch(mat_file=Test_mat_file)._read_mat()
    train_loader=DataLoader(train_OE_Data,batch_size=200,shuffle=True,drop_last=True)
    
    #两个网络
    fno=FNO(n_modes=(64,64),hidden_channels=64,in_channels=2,out_channels=2)
    bran_nn=branch_net(2,50,1)
    mse=torch.nn.MSELoss()
    optimzer1=torch.optim.Adam(fno.parameters(),lr=0.001)
    optimzer2=torch.optim.Adam(bran_nn.parameters(),lr=0.001)
   
    num_epochs=100
   
    losses = np.zeros(num_epochs)  # 提前分配内存空间
    test_losses=np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
       
      for i,(data,next_t,condition) in enumerate(train_loader):

         expand_size=data.shape[-2] #300 =100*3
   
         out=bran_nn(condition)
         out=out.unsqueeze(-2)
         out=out.repeat(1,1,expand_size,1)
         fno_out=fno(data) #fno 输出的是下一个时间步的数据
         #两个网络的输出相乘
   
         final=fno_out*out


         true=next_t[:,:,:].float()
         loss=mse(final,true)
         optimzer1.zero_grad()
         optimzer2.zero_grad()
         loss.backward()
         optimzer1.step()
         optimzer2.step()
         
      losses[epoch] = loss.item()
      
      if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
     
      
      if epoch % 1 == 0:
         
         print(f"epoch:{epoch},loss:{loss}")
      
         test_loss=test(
                        folder=model_save_path,
                        epoch=epoch,
                        test_data=test_np_data,
                        condition=test_OE_Data.conditions_to_tensor())
         test_losses[epoch]= test_loss
         
      if epoch % 100 == 0 and epoch != 0:
         torch.save(fno.state_dict(), f"{model_save_path}/fno_{epoch}.pth")
         torch.save(bran_nn.state_dict(), f"{model_save_path}/bran_nn_{epoch}.pth")
         
   # 保存训练损失和测试损失 npz
    np.savez(f"{model_save_path}/losses.npz",
            training_loss=losses, test_loss=test_losses)
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label="Training loss", linewidth=3)
    plt.plot(test_losses, label="Test loss", linewidth=3)
    plt.legend()
    plt.yscale('log')
    plt.savefig(f"{model_save_path}/losses.png")
    
    plt.close()
         
  
         
            
    

    