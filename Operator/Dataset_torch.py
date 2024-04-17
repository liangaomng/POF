
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import re
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入tqdm
import os

def Get_4_folder(operation,type,name="NCHD",**kwargs):
    '''
    kwargs : torch_data,Conditions
    operations :str,load or save
    type:str,train or test
    '''

    if operation == "load":
        folder_name = f'Data/{name}' + '_pt'
        file_name = f'{type}_data.pt'
        file_path = os.path.join(folder_name, file_name)
        if os.path.isfile(file_path):
            print(f"The file {file_name} exists in {folder_name}.")
            data = torch.load(file_path)
            dataset = data[f'{type}_dataset']
            torch_data = dataset["torch_data"]
        
            return torch_data
        else:
            assert "The file {file_name} does not exist in {folder_name}."
    
    elif operation == "save":
        torch_data = kwargs.get("torch_data")
        Conditions = kwargs.get("Conditions")

        if torch_data is None or Conditions is None:
            assert "Missing data: 'torch_data' or 'Conditions' not provided."

        folder_name = f'Data/{name}' + '_pt'
        file_name = f'{type}_data.pt'
        file_path = os.path.join(folder_name, file_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Saving only the necessary data
        data_to_save = {
            f'{type}_dataset': {
                "torch_data": torch_data
            }
        }
        torch.save(data_to_save, file_path)
        print(f"The file {file_name} saved in {folder_name}.")

  




class OE_Dataset(Dataset):
    def __init__(self, input_data,input_cond,out,task="NCHD",**kwargs):
      

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_data = input_data.to(self.device) #[b, 1,1, 300] 时间维度已经在前面
        self.condition = input_cond
        self.ground_truth = out.to(self.device)# ground truth
        self.task = task
         #这里condition的是A和L 两个条件
        self.con_tensor=self.conditions_to_tensor()
    
       
        
    def __len__(self):
        
        return  self.input_data.shape[0]

    def __getitem__(self, idx):
    
       
        ini_data = self.input_data[idx] # ini_eta 
        ground_truth = self.ground_truth[idx]
        condition = self.con_tensor[idx] # condition,已经进入cuda
   
        
        return ini_data,condition,ground_truth
    
    def conditions_to_tensor(self):
       # 假设每个条件都有2个值（A和L）
        if self.task =="NCHD":# 2个条件
            
            tensor = torch.zeros((len(self.condition), 2), dtype=torch.float32).to(self.device)
       
        
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
        self.datas = None  # 初始化datas为None
        for i,mat in enumerate(mat_file):
            pattern = r'A_([0-9.]+)_L_([0-9.]+)'

            # 搜索模式并提取数字
            match = re.search(pattern, mat)
            self.A = np.float32(match.group(1))
            self.L = np.float32(match.group(2))
            self.CONDITIONS[f"{mat}"]={ "A":self.A,
                                        "L":self.L}
        
    def _return_conditions(self):
        return self.CONDITIONS 

    def _read_mat(self):
        
        files = len(self.mat_file)
        self.datas= np.zeros((files,640,300,3))
        
        for i, mat in tqdm(enumerate(self.mat_file), total=len(self.mat_file), desc="Loading MAT files"):
            
            self.data = scipy.io.loadmat(mat)
            self.wave_data = self.data['wave_data'][0, 0]

            title = ["deepsea", "slope", "normal"]

            self.deepsea_data = self.wave_data['deepsea'].reshape(640, 100, 3).astype(np.float32)
            self.slope_data = self.wave_data['slope'].reshape(640, 100, 3).astype(np.float32)
            self.normal_data = self.wave_data['normal'].reshape(640, 100, 3).astype(np.float32)

            current_data = np.concatenate((self.deepsea_data, self.slope_data, self.normal_data), axis=1)
            self.datas[i] = current_data
            
            #tqdm.write(f"Processed {i+1}/{len(self.mat_file)} files")

        self.datas = torch.from_numpy(self.datas).float() #转torch
        self.datas = self.datas.permute(0,3,1,2) # 
        print("data shape",self.datas.shape)
  
        return self.datas,self.CONDITIONS

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


            

           
            
            
        
        

    
