# An-algorithm-for-blurring-clear-images
运行环境:Kaggle 具体网址https://www.kaggle.com/ 注册需要魔法，使用时不需要  
zxb的比赛模型，所选用的数据集为kaggle中realblurj-image-deblurring-dataset，该数据集比Real_Blur_Dataset训练效果更好  
blind-unet 最终PSNR为19.21  
blind-NBFA 最终PSNR为22.03  


盲卷积模块知识。  
定义  
盲卷积是在不知道模糊核（Point Spread Function，PSF）信息的情况下，通过算法从模糊图像中恢复出清晰图像的过程。传统去卷积方法需要已知模糊核，而盲卷积无需先验知识，能同时估计模糊核和恢复原始图像。  
图像模型:
盲卷积基于一个图像模型，假设模糊图像是清晰图像和一个未知的卷积核(模糊核)的卷积结果，同时可能包含一些噪声。数学表达式为:y=h*x+n。  
其中:
-y 是模糊图像(已知)。  
-x 是清晰图像(未知，要恢复的)。  
-h 是卷积核(未知，要估计的)。  
-n 是噪声(通常假设为高斯白噪声)。  
![image](https://github.com/angelandeagle/An-algorithm-for-blurring-clear-images/blob/main/%E5%8E%BB%E7%87%A5%E7%9F%A5%E8%AF%86/image.png)
