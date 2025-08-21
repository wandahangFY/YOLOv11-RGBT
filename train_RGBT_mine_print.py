import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import RTDETR

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')
    # model = YOLO('runs/dota8/dota8-yolo11n-RGBT-midfusion-obb-e300-16-27/weights/best.pt')

    model.info(True,True)
    # model.load('yolov8n.pt') # loading pretrain weights
    # model.train(data=R'ultralytics/cfg/datasets/M3FD.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=300,
    #             batch=32,
    #             close_mosaic=0,
    #             workers=2,
    #             device='0',
    #             optimizer='SGD',  # using SGD
    #             # resume='', # last.pt path
    #             # amp=False, # close amp
    #             # fraction=0.2,
    #             use_simotm="RGBT",
    #             channels=4,
    #             project='runs/M3FD',
    #             name='M3FD-yolo11-RGBT-midfusion-CAS-',
    #             )