import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R"PVELAD/PVELAD-yolov8n-DBBNCSPELAN12/weights/best.pt") # select your model.pt path
    model.predict(source=r'G:\wan\data\PVELAD_C\good_corner_png2',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  show=False,
                  save_frames=True,
                  use_simotm="Gray16bit",
                  channels=1,
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )