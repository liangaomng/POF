import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def calculate_psnr(image1, image2):
    # 计算MSE
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr,mse

def calculate_ssim(image1, image2):
    # 确保图像为灰度图
    if image1.shape[-1] == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if image2.shape[-1] == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    ssim = compare_ssim(image1, image2,data_range=1)
    return ssim