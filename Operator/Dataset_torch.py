
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import re
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class OE_Dataset(Dataset):
    def __init__(self, data,condition,transformer=None):
        self.data = data
        self.condition=condition#{'Data/A_0.3_L_2.0_wave_data.mat': {'A': 0.3, 'L': 2.0}, 'Data/A_0.3_L_5.0_wave_data.mat': {'A': 0.3, 'L': 5.0}}
        self.transform=transformer
    def __len__(self):
        # 数据集大小为总时间步数减一，因为我们总是预测下一个时间步
        return self.data.shape[1] - 1

    def __getitem__(self, idx):
        # 返回当前时间步和下一个时间步的数据和condition 
        #这里condition的是A和L $#注意data (n, 640, 300, 3)
        self.con_tensor=self.conditions_to_tensor()
        #print("con_tensor",self.con_tensor.shape)  idx是时间维度
        if self.transform:
            print("bianhuan")
            self.data[:,idx,:,:] = self.transform(self.data[:,idx,:,:])
            self.data[:,idx + 1,:,:] = self.transform(self.data[:,idx + 1,:,:])
        return self.data[:,idx,:,:], self.data[:,idx + 1,:,:],self.con_tensor[:,:]
    
    def conditions_to_tensor(self):
       # 假设每个条件都有2个值（A和L）
        tensor = torch.zeros((len(self.condition), 2), dtype=torch.float32)
        
        for i, (key, value) in enumerate(self.condition.items()):
            tensor[i, 0] = torch.tensor(value['A'], dtype=torch.float32)  # 第一个值是A
            tensor[i, 1] = torch.tensor(value['L'], dtype=torch.float32)  # 第二个值是L

        return tensor

# 自定义归一化变换
class NormalizeTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        return (x - self.mean) / self.std
    
    def inverse(self, x):
        return x * self.std + self.mean


class Read_Mat_4torch():
    def __init__(self, mat_file:list):
        
        self.mat_file = mat_file
        # 正则表达式模式
        self.CONDITIONS = {}
        for i,mat in enumerate(mat_file):
            pattern = r'A_([0-9.]+)_L_([0-9.]+)'

            # 搜索模式并提取数字
            match = re.search(pattern, mat)
            self.A = np.float32(match.group(1))
            self.L = np.float32(match.group(2))
            self.CONDITIONS[f"{mat}"]={ "A":self.A,
                                        "L":self.L}
        print(self.CONDITIONS)

    def _read_mat(self):
        
        for i,mat in enumerate(self.mat_file):
            self.data = scipy.io.loadmat(mat)
            #(640,100,3)
            self.wave_data = self.data['wave_data'][0, 0]
            # 有三段
            title=["deepsea","slope","normal"]
            #时间，空间，3
            self.deepsea_data = self.wave_data['deepsea'].reshape(640,100,3).astype(np.float32)
            self.slope_data =  self.wave_data['slope'].reshape(640,100,3).astype(np.float32)
            self.normal_data =  self.wave_data['normal'].reshape(640,100,3).astype(np.float32)
            #把三段数据拼接起来
            #datas (640, 300, 3)
            self.datas=np.concatenate((self.deepsea_data,self.slope_data,self.normal_data),axis=1)
            # 不同的文件的维度是[n,640.300,3]
            
            if i==0:
                self.datas=self.datas
                
            else:
                self.datas=np.concatenate((self.datas,self.datas),axis=0)
                self.datas=self.datas.reshape(-1,640,300,3)
            print("after concat",self.datas.shape)
        # 假设 train_data 是你的训练数据
        mean = self.datas.mean()
        std = self.datas.std()
        # 创建归一化变换实例
        normalize_transform = NormalizeTransform(mean, std)
        
        OE_Data=OE_Dataset(self.datas,self.CONDITIONS)
        return self.datas,OE_Data

class branch_net(nn.Module):
    def __init__(self,input,hidden,output):
        super(branch_net,self).__init__()
            
        self._net = nn.Sequential(nn.Linear(input, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, output),
                    )
    def forward(self,x):
        out=self._net(x)
        return out
        
from neuralop.models import FNO       
        
if __name__ == "__main__":
    # 加载 .mat 文件
    mat_file = ['Data/A_0.3_L_2.0_wave_data.mat','Data/A_0.3_L_5.0_wave_data.mat']
    
    
    np_data,OE_Data=Read_Mat_4torch(mat_file)._read_mat()
    DataLoader=DataLoader(OE_Data,batch_size=20,shuffle=True)
    fno=FNO(n_modes=(16,16),hidden_channels=12,in_channels=2,out_channels=2)
    mse=torch.nn.MSELoss()
    optimze=torch.optim.Adam(fno.parameters(),lr=0.001)
    bran_out=branch_net(2,50,1)
    for epoch in range(10):
        for i,(data,next_t,condition) in enumerate(DataLoader):
            data=data[:,:,:,:]
            condition=condition
            expand_size=data.shape[-2]
    
            out=bran_out(condition)
            out=out.unsqueeze(-2)
            out=out.repeat(1,1,300,1)


            fno_out=fno(data)
            print("data",data.shape)

            final=fno_out*out
            true=next_t[:,:,:].float()
            loss=mse(final,true)
            optimze.zero_grad()
            loss.backward()
            optimze.step()
        print(f"epoch:{epoch},loss:{loss}")
            
    np_data=np_data[:,:,:,:]
    print(np_data.shape)
    # 测试
    test=np.zeros_like(np_data)
    with torch.no_grad():
        for i in range(639):
            fnp_out=fno(torch.tensor(np_data[:,i:i+2,:,:], dtype=torch.float32))

            out=bran_out(OE_Data.conditions_to_tensor())
            out=out.unsqueeze(-2)
            out=out.repeat(1,1,300,1)
            #([2, 2, 300, 3])
            test[:,i:i+2,:,:]=(fnp_out*out).cpu().numpy()
    print(test.shape)
    fig,ax= plt.subplots(1,2,figsize=(10,8))
    ax[1].imshow(test[0,:,:,2],cmap='jet',vmin=0, vmax=0.5)
    ax[1].set_title("predict")
    ax[0].imshow(np_data[0,:,:,2],cmap='jet',vmin=0, vmax=0.5)
    ax[0].set_title("true")
    plt.show()
            
    

            

           
            
            
        
        

    
