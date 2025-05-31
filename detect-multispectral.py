import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"runs/coco8-multispectral/coco8-multispectral-yolo11-RGBT-earlyfusion-10c-16/weights/best.pt") # select your model.pt path
    model.predict(source=r'G:\wan\code\GitPro\YOLOv11-RGBT\YOLOv11-RGBT\datasets\coco8-multispectral\images\train',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  show=False,
                  save_frames=True,
                  use_simotm="Multispectral",
                  channels=10, # 任意
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )