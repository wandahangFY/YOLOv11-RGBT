import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion.yaml')
    # model.info(True,True)
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/LLVIP-rgbt.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # lr0=0.002,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                # pairs_rgb_ir=['visible','infrared'] , # default: ['visible','infrared'] , others: ['rgb', 'ir'],  ['images', 'images_ir'], ['images', 'image']
                use_simotm="RGBT",
                channels=4,
                project='runs/LLVIP',
                name='LLVIP-yolo11n-RGBT-midfusion-',
                )