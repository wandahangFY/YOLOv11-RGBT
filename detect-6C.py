import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"runs/LLVIP_r20/LLVIP_r20-yolov8-RGBRGB6C-midfusion/weights/best.pt") # select your model.pt path
    model.predict(source=r'E:\BaiduNetdiskDownload\RGB_IF\LLVIP\LLVIP\images\visible\trainval',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  show=False,
                  save_frames=True,
                  use_simotm="RGBRGB6C",
                  channels=6,
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )