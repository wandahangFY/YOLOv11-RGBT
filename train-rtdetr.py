import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr-RGBT/rtdetr-resnet50-RGBT-midfusion-P3.yaml')
    # model.load('') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/LLVIP_r20.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=2,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBT",
                channels=4,
                project='runs/LLVIP_r20',
                name='LLVIP_r20-rtdetr-resnet50-RGBT-midfusion-P3-',
                )