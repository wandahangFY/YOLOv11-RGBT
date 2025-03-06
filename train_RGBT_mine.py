import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-EFM.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/M3FD.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=0,
                workers=4,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBT",
                channels=4,
                project='runs/M3FD',
                name='M3FD-yolo11n-RGBT-midfusion-EFM-e300-16-',
                )