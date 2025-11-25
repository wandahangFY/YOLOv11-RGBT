import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11-cls-RGBRGB.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r"datasets/cifar10",
                cache=False,
                imgsz=32,
                epochs=100,
                batch=512,
                workers=4,
                # device='0',
                seed=0,
                pretrained=False,
                optimizer='SGD',  # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBRGB6C",
                channels=6,
                project='runs/train/cifar10',
                name='yolov11n-cls-cifar10-seed-RGBRGB6C',
                )


    # model = YOLO('ultralytics/cfg/models/11/yolo11-cls.yaml')
    # # model.load('yolov8n.pt') # loading pretrain weights
    # model.train(data=r"datasets/cifar10/visible",
    #             cache=False,
    #             imgsz=32,
    #             epochs=100,
    #             batch=512,
    #             workers=4,
    #             # device='0',
    #             seed=0,
    #             pretrained=False,
    #             optimizer='SGD',  # using SGD
    #             # patience=0, # close earlystop
    #             # resume=True, # 断点续训,YOLO初始化时选择last.pt
    #             # amp=False, # close amp
    #             # fraction=0.2,
    #             use_simotm="RGB",
    #             channels=3,
    #             project='runs/train/cifar10',
    #             name='yolov11n-cls-cifar10-seed-visible',
    #             )
    #
    #
    # model = YOLO('ultralytics/cfg/models/11/yolo11-cls.yaml')
    # # model.load('yolov8n.pt') # loading pretrain weights
    # model.train(data=r"datasets/cifar10/infrared",
    #             cache=False,
    #             imgsz=32,
    #             epochs=100,
    #             batch=512,
    #             workers=4,
    #             # device='0',
    #             seed=0,
    #             pretrained=False,
    #             optimizer='SGD',  # using SGD
    #             # patience=0, # close earlystop
    #             # resume=True, # 断点续训,YOLO初始化时选择last.pt
    #             # amp=False, # close amp
    #             # fraction=0.2,
    #             use_simotm="RGB",
    #             channels=3,
    #             project='runs/train/cifar10',
    #             name='yolov11n-cls-cifar10-seed-infrared',
    #             )