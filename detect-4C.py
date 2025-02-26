import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R"G:\wan\code\GitPro\ultralytics-8.2.79\PVELAD\PVELAD-yolov10-RGBT-midfusion\weights\best.pt") # select your model.pt path
    model.predict(source=r'E:\BaiduNetdiskDownload\RGB_IF\LLVIP\LLVIP\images\visible\trainval',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  show=False,
                  save_frames=True,
                  use_simotm="RGBT",
                  channels=4,
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )