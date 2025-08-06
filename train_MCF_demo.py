import warnings

import torch

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    '''
    后续会探索更为简单的训练方式，暂时先利用下列方式进行MCF训练
    
        Step 1. 加载预训练权重,或者利用其他方式,得到一个 单模态或者单光谱效果比较好的检测模型，当做主分支冻结部分的权重,用于第3步的网络权重转换（建议红外和可将光都训练一个，用于确定主分支）
        Step 2. 设置   epochs=1,  fraction=0.01, # 仅用  训练一个随机初始化的网络权重,用于第3步的网络权重转换
        Step 3. 将第一步得到的模型权重加载到第二步的网络结构中，并将 ZeroConv2d 部分的权重清零，得到 yolo11n-RGBT-midfussion-MCF.pt
        Step  4. 将第三步得到的模型直接用于训练，不要加载yaml，直接加载   yolo11n-RGBT-midfussion-MCF.pt 文件 进行训练
    
    
    We will later explore simpler training methods. For now, we will conduct MCF training using the following approach. 
        Step 1. Load the pre-trained weights, or use other methods to obtain a detection model with better single-modal or single-spectrum performance, and use the weights of the frozen part of the main branch as the weights for the network weight conversion in the third step (it is recommended to train both infrared and visible light separately for this purpose, to determine the main branch).
        Step 2. Set epochs = 1, fraction = 0.01. # Only use to train a randomly initialized network weight for the network weight conversion in the third step.
        Step 3. Load the model weights obtained in the first step into the network structure of the second step, and clear the weights of the ZeroConv2d part, obtaining yolo11n-RGBT-midfussion-MCF.pt.
        Step 4. Use the model obtained in the third step directly for training, do not load the yaml file, and directly load the yolo11n-RGBT-midfussion-MCF.pt file for training.
    
    [![Google Drive Models & Datasets](https://drive.google.com/drive/folders/14T2OaLAiMxlx8WJVyJ2x5DLI8RNI0R8m?usp=drive_link) 
    [![Baidu Drive Models](https://pan.baidu.com/s/1Q6H98fiW_f7Kdq6-Ms6oUg?pwd=669j) 
    [![Baidu Drive Datasets](https://pan.baidu.com/s/1xOUP6UTQMXwgErMASPLj2A?pwd=9rrf)

    '''

    # Step 1

    # model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    # model.train(data=R'ultralytics/cfg/datasets/M3FD.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=300,
    #             batch=16,
    #             close_mosaic=0,
    #             workers=2,
    #             device='0',
    #             optimizer='SGD',  # using SGD
    #             # resume='', # last.pt path
    #             # amp=False, # close amp
    #             use_simotm="RGB",
    #             channels=3,
    #             project='runs/M3FD',
    #             name='M3FD-yolo11n-RGB-',
    #             )
    # del model
    # torch.cuda.empty_cache()



    # Step 2
    model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-MCF.yaml')
    model.train(data=R'ultralytics/cfg/datasets/M3FD.yaml',
                cache=False,
                imgsz=640,
                epochs=1,
                batch=16,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                fraction=0.01,  # 仅用 1% 的数据训练, 快速得到一个模型权重模板    Train with only 1% of the data. Quickly obtain a model weight template
                use_simotm="RGBRGB6C",
                channels=6,
                project='runs/M3FD',
                name='M3FD-yolo11n-RGBT-midfusion-MCF-e300-16-',
                )
    del model
    torch.cuda.empty_cache()

    # Step 3     python transform_MCF.py


    # Step 4

    # model = YOLO(r'M3FD-yolo11n-RGBT-midfusion-MCF.pt')
    # model.train(data=R'ultralytics/cfg/datasets/M3FD.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=100,
    #             batch=16,
    #             close_mosaic=0,
    #             workers=2,
    #             device='0',
    #             optimizer='SGD',  # using SGD  微调参数请参考论文进行设置，事实上，论文中的参数大概率也不是最佳参数，我们对超参数的选取没有做大量测试。仅做了几组可行的参数设置
    #             # For fine-tuning the parameters, please refer to the paper for setting. In fact, the parameters in the paper are probably not the optimal ones. We did not conduct extensive tests on the selection of hyperparameters. We only made a few sets of feasible parameter settings.
    #
    #             # resume='', # last.pt path
    #             # amp=False, # close amp
    #             # fraction=0.2,
    #             freeze=[2, 3, 4, 5, 6, 17, 18, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    #             use_simotm="RGBRGB6C",
    #             channels=6,
    #             project='runs/M3FD',
    #             name='M3FD-yolo11n-RGBT-midfusion-MCF-final-e300-16-',
    #             )
    # del model
    # torch.cuda.empty_cache()

