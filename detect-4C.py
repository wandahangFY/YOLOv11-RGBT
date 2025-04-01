import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R"runs/M3FD/M3FD-yolov5-RGBT-midfusion/weights/best.pt") # select your model.pt path
    model.predict(source=r"G:\wan\data\RGBT\testVD\visible\video.mp4",
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