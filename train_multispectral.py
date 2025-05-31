import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11-RGBT-earlyfusion-10c.yaml')
    # model.info(True,True)
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/coco8-multispectral.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # lr0=0.002,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="Multispectral",
                channels=10,
                project='runs/coco8-multispectral',
                name='coco8-multispectral-yolo11-RGBT-earlyfusion-10c-',
                )