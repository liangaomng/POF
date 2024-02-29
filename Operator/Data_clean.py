import os


#查找该目录里面所有.mat 文件 返回name list
def return_name_list(folder):
    # 初始化一个空列表，用于存储找到的.mat文件名
    name_list = []
    
    # os.walk遍历文件夹
    for dirpath, dirnames, filenames in os.walk(folder):
        # 遍历文件名
        for filename in filenames:
            # 检查文件扩展名是否为.mat
            if filename.endswith('.mat'):
                # 如果是，将文件名添加到列表中
                name_list.append(folder+"/"+filename)
                
    return name_list

   