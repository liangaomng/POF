
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
from torchvision import transforms



class OE_Dataset(Dataset):
    def __init__(self, data,condition,**kwargs):
      
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data = torch.tensor(data,dtype=torch.float32).to(device)
        self.condition=condition
         # 实例化NormalizeTransform类
        self.norm_transform = NormalizeTransform(mean=kwargs['mean'], 
                                                 std=kwargs['std'])
        self.task =kwargs["task"]
    def __len__(self):
        
        return  self.data.shape[0]

    def __getitem__(self, idx):
    
        #这里condition的是A和L $#注意data (n_files, 640, 300, 3) 
        self.con_tensor=self.conditions_to_tensor()
        # #对第三个通道-解进行归一化
        # data=self.norm_transform (self.data[:,:,:,2])
        # #放回
        # self.data[:,:,:,2]=data

        #permute 成【n_files, 3, 640, 300】
        permute_data=self.data.permute(0,3,1,2)
        return permute_data[idx,:,:,:], self.con_tensor[idx,:]
    
    def conditions_to_tensor(self):
       # 假设每个条件都有2个值（A和L）
        if self.task =="OE":#3 个条件
            tensor = torch.zeros((len(self.condition), 3), dtype=torch.float32)
        else:
            tensor = torch.zeros((len(self.condition), 2), dtype=torch.float32)
        
        for i, (key, value) in enumerate(self.condition.items()):
            tensor[i, 0] = torch.tensor(value['A'], dtype=torch.float32)  # 第一个值是A
            tensor[i, 1] = torch.tensor(value['L'], dtype=torch.float32)  # 第二个值是L
            if self.task =="OE":
                tensor[i, 2] = torch.tensor(value['dr'], dtype=torch.float32)  # 第二个值是L

        return tensor
    def Get_transform(self): 
        return self.norm_transform

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
        self.datas = None  # 初始化datas为None
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
            print("i",i)
            self.data = scipy.io.loadmat(mat)
            #(640,100,3)
            self.wave_data = self.data['wave_data'][0, 0]
            # 有三段
            title=["deepsea","slope","normal"]
            #时间，空间，3
            self.deepsea_data = self.wave_data['deepsea'].reshape(640,100,3).astype(np.float32)
            self.slope_data =  self.wave_data['slope'].reshape(640,100,3).astype(np.float32)
            self.normal_data =  self.wave_data['normal'].reshape(640,100,3).astype(np.float32)
       
            current_data=np.concatenate((self.deepsea_data,self.slope_data,self.normal_data),axis=1)

            current_data=current_data.reshape(-1,640,300,3) 

            # 不同的文件的维度是[n,640,300,3] 3 是3 个同搭配
            
            if i==0:
                self.datas=current_data
                
            else:
                print("test", self.datas.shape)
                
                self.datas=np.concatenate((self.datas,current_data),axis=0)
                self.datas=self.datas.reshape(-1,640,300,3)
                

            
        # dim=[0, 2, 3]告诉PyTorch沿着批次（0维）、高度（2维）、宽度（3维）维
        #返回训练的三个通道（解）的均值和标准差
        mean = self.datas.mean(axis=(0, 1, 2))
        std = self.datas.std(axis=(0, 1, 2))
      
        
        OE_Data=OE_Dataset(self.datas,self.CONDITIONS,mean=mean[-1],std=std[-1])
        norm_transform=OE_Data.Get_transform()
        return self.datas,OE_Data,norm_transform

class Read_OE_Mat_4torch():
    def __init__(self, mat_file:list):
        
        self.mat_file = mat_file
        # 正则表达式模式
        self.CONDITIONS = {}
        self.datas = None  # 初始化datas为None
        
        for i,mat in enumerate(mat_file):
            print(mat)
            pattern = r'A_([0-9.]+)_L_([0-9.]+)_dr_([0-9.]+)'
            # 搜索模式并提取数字
            match = re.search(pattern, mat)
            self.A = np.float32(match.group(1))
            self.L = np.float32(match.group(2))
            self.dr = np.float32(match.group(3))
            self.CONDITIONS[f"{mat}"]={ "A":self.A,
                                        "L":self.L,
                                        "dr":self.dr}
        print(self.CONDITIONS)

    def _read_mat(self):
        
        for i,mat in enumerate(self.mat_file):
            print("i",i)
            self.data = scipy.io.loadmat(mat)
            #(640,100,3)
            self.wave_data = self.data['wave_data'][0, 0]
            # 有三段
            title=["deepsea","slope","normal"]
            #时间，空间，3
            self.deepsea_data = self.wave_data['deepsea'].reshape(640,100,3).astype(np.float32)
            self.slope_data =  self.wave_data['slope'].reshape(640,100,3).astype(np.float32)
            self.normal_data =  self.wave_data['normal'].reshape(640,100,3).astype(np.float32)
       
            current_data=np.concatenate((self.deepsea_data,self.slope_data,self.normal_data),axis=1)

            current_data=current_data.reshape(-1,640,300,3) 

            # 不同的文件的维度是[n,640,300,3] 3 是3 个同搭配
            
            if i==0:
                self.datas=current_data
                
            else:
                print("test", self.datas.shape)
                
                self.datas=np.concatenate((self.datas,current_data),axis=0)
                self.datas=self.datas.reshape(-1,640,300,3)
                

            
        # dim=[0, 2, 3]告诉PyTorch沿着批次（0维）、高度（2维）、宽度（3维）维
        #返回训练的三个通道（解）的均值和标准差
        mean = self.datas.mean(axis=(0, 1, 2))
        std = self.datas.std(axis=(0, 1, 2))
      
        
        OE_Data=OE_Dataset(self.datas,self.CONDITIONS,mean=mean[-1],std=std[-1],task="OE")
        norm_transform=OE_Data.Get_transform()
        return self.datas,OE_Data,norm_transform
class branch_net(nn.Module):
    def __init__(self,input,hidden,output):
        super(branch_net,self).__init__()
            
        self._net = nn.Sequential(nn.Linear(input, hidden),
                    nn.tanh(),
                    nn.Linear(hidden, hidden),
                    nn.tanh(),
                    nn.Linear(hidden, output),
                    )
    def forward(self,x):
        out=self._net(x)
        return out


            

           
            
            
        
        

    
