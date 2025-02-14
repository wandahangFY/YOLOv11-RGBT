import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8-RGBT/yolov8n-RGBRGB6C-midfusion.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/LLVIP_r20.yaml',
                cache=False,
                imgsz=640,
                epochs=20,
                batch=16,
                close_mosaic=10,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBRGB6C",
                channels=6,
                project='PVELAD',
                name='PVELAD-yolov8-RGBRGB6C-midfusion',
                )