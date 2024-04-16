import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from abc import abstractmethod
print("hi")

class Metric():
    
    @classmethod
    def calculate_psnr(cls,image1, image2)->float:
        # 计算psnr
        mse = np.mean((image1 - image2) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 10 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    @classmethod
    def calculate_mse(cls,image1, image2)->float:
        # 计算mse
        mse = np.mean((image1 - image2) ** 2)
        return mse
    
    
    @classmethod
    def calculate_ssim(cls,image1, image2)->float:
        # 确保图像为灰度图
        if image1.shape[-1] == 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        if image2.shape[-1] == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        ssim = compare_ssim(image1, image2,data_range=1)
        return ssim
    
class Global():
    '''
    Global Performance
    '''
    def __init__(self):
        super().__init__()
        self.__Global_dict={
                    "psnr":[], #with different imgs
                    "ssim":[],
                    "mse":[],
                    "notes":"Global performance"}  

    @property  
    def Global_dict(self):
        return self.__Global_dict

    @Global_dict.setter
    def Global_dict(self,note:str)->None:
        self.__Global_dict["notes"]=note

    def Calulate(self, img1, img2,choose="global")->None:
        
        image_numbers = img1.shape[0]
        test_imgs_psnr = []
        test_imgs_ssim = []
        test_imgs_mse = []

        if choose=="global":
            
            for i in range(image_numbers):
                #这里有小问题，是不是每个都算了
                test_imgs_psnr.append ( Metric.calculate_psnr(img1[i,:,:], img2[i,:,:]))
                test_imgs_ssim.append ( Metric.calculate_ssim(img1[i,:,:], img2[i,:,:]))
                test_imgs_mse.append ( Metric.calculate_mse(img1[i,:,:], img2[i,:,:]))
                
            self.__Global_dict["psnr"] .append(np.mean(test_imgs_psnr))
            self.__Global_dict["ssim"]. append(np.mean(test_imgs_ssim))
            self.__Global_dict["mse"] .append(np.mean(test_imgs_mse))

        elif choose=="local":
                assert False, "Not implemented"

class Local():
    '''
    Local Performance
    '''
    def __init__(self):
        super().__init__()
        self.__Local_dict={
                    "psnr":[],
                    "ssim":[],
                    "mse":[],
                    "notes":"Lobal performance"}  

    @property  
    def Local_dict(self):
        return self.__Local_dict

    @Local_dict.setter
    def Local_dict(self,note:str)->None:
        self.__Local_dict["notes"]=note

    def Calulate(self, img1, img2,choose="local")->None:
        
        image_numbers = img1.shape[0]
        test_imgs_psnr = []
        test_imgs_ssim = []
        test_imgs_mse = []
        print("imag1",img1.shape)
        if choose=="local":
            for i in range(image_numbers):
                test_imgs_psnr.append ( Metric.calculate_psnr(img1[i,:,:], img2[i,:,:]))
                test_imgs_ssim.append (Metric.calculate_ssim(img1[i,:,:], img2[i,:,:]))
                test_imgs_mse.append ( Metric.calculate_mse(img1[i,:,:], img2[i,:,:]))
            self. __Local_dict["psnr"].append(np.mean(test_imgs_psnr))
            self.__Local_dict["ssim"].append(np.mean(test_imgs_ssim))
            self.__Local_dict["mse"].append(np.mean(test_imgs_mse))
            
            
        elif choose=="global":
                assert False, "Not implemented"            

global_metric = Global()
local_metric = Local()

if __name__=="__main__":
    
    global_metric=Global()
    
    


    
    
