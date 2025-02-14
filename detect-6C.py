import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"G:\wan\code\GitPro\ultralytics-8.2.79\PVELAD\PVELAD-yolov8-RGBRGB6C-midfusion\weights\best.pt") # select your model.pt path
    model.predict(source=r'E:\BaiduNetdiskDownload\RGB_IF\LLVIP\LLVIP\images\visible\trainval',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  show=True,
                  save_frames=True,
                  use_simotm="RGBRGB6C",
                  channels=6,
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )