
import torch 
import torch.nn as nn
from neuralop.models import FNO,UNO
import sys
sys.path.append('../OE')
import sys
import os
import numpy as np
absolute_path = os.path.abspath(sys.path[-1])
print("添加的绝对路径是:", absolute_path)
# 获取最后添加的路径的绝对路径
from Operator import utilities3
from Operator.Dataset_torch import Read_Mat_4torch,OE_Dataset,Get_4_folder
from Operator.Data_clean import return_name_list
from torch.utils.data import DataLoader
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
  
  def __init__(self,infeature=2,Trunk="FNO",
               t_steps=640,wavelet="db6",fno_modes=64):
      super(MFNO,self).__init__()
      self.bran_nn = Branch_net(infeature,50,1)
      self.Trunk = Trunk
      if self.Trunk  == "FNO":
        #modes64 的参数量 2605052
        print("FNO")
        self.trunk_nn = FNO(n_modes=(fno_modes,fno_modes),
                            hidden_channels=12,in_channels=1,out_channels=t_steps)
        
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
class ExclusivityLoss(nn.Module):
    def __init__(self):
        super(ExclusivityLoss, self).__init__()

    def forward(self, outputs):
        # 按输出值排序，然后计算排序后相邻输出之间的差异
        sorted_outputs, indices = torch.sort(outputs.view(-1))
        differences = sorted_outputs[1:] - sorted_outputs[:-1]
        # 惩罚相邻输出之间差异小的情况
        
        loss = -torch.log(differences + 1e-12).mean()  # 加入小的常数避免对0取对数
        return loss
    
class Fine_tune():

    def __init__(self,net_path):

        self.ini_net_dict = torch.load(net_path)
        self.net = MFNO().to("cuda")
        self.net.load_state_dict(self.ini_net_dict)
        #优化
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 1e-3)
        self.loss_func = ExclusivityLoss()
        self.mse = nn.MSELoss()


    def train(self,train_loader,epoch):
        print("tune",flush=True)
        for epoch in range(epoch):
            for i, (ini_data, condition,ground_truth) in enumerate(train_loader):
                ini_data = ini_data #[batch,1,1,300] first1 is time
                condition = condition#[batch,2(A and L)]
                ground_truth = ground_truth
        
                final_output = self.net(ini_data,condition)
                br_out = self.net.bran_nn(condition)
                # 增大branch net的单射 损失
                loss1 =  self.loss_func(br_out) 
                loss2 =  self.mse(final_output,ground_truth)
                loss = 10*loss1 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch % 5 == 0:
                print('Epoch: ', epoch, '| loss1 : ',loss1.item(),'| loss2 : ',loss2.item())
        return self.net
    
def prepare(data_folder,batch_size):

    All_mat_file= return_name_list(folder = data_folder)
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
    print("before_norm",train_torch_data.shape,flush=True) #norm torch.Size([b, 3,640, 300]) 

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
    
    # #保存
    # torch.save(train_OE_Dataset, 'Data/NCHD_pt/train_OE_Dataset.pt')
    # torch.save(test_OE_Dataset, 'Data/NCHD_pt/test_OE_Dataset.pt')
  

    #定义模型和相关优化器
    train_loader = DataLoader(train_OE_Dataset,batch_size = batch_size,shuffle=True)
    test_loader = DataLoader(test_OE_Dataset,batch_size = batch_size,shuffle=True)
    return train_loader,test_loader
        
if __name__=="__main__":
    print("hi")

    args={"data_folder":"/liujinxin/lam/OE/Data/Train/NCHD",
          "batch_size":50
          }
    fine_tune = Fine_tune( net_path = "Inverse/mno_ckpt.pth")
    train_loader,test_loader = prepare(data_folder = args["data_folder"],
                                       batch_size=args["batch_size"])
    

    fine_tune.train(train_loader,epoch=2000)
    torch.save(fine_tune.net.state_dict(), 'fine_tune_mno.pth')

    

