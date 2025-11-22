import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    '''
        和原始YOLOv11的detect相比，多了两个参数
            use_simotm="Gray",
            channels=1,

        Compared with the original "detect" in YOLOv11, it now has two additional parameters: 
            "use_simotm" set to "Gray". channels=1,
    '''

    model = YOLO(R"PVELAD/PVELAD-yolov8n/weights/best.pt") # select your model.pt path
    model.predict(source=r'G:\wan\data\PVELAD\good_corner_png2',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  show=False,
                  save_frames=True,
                  use_simotm="Gray", # Gray: uint8  Gray16bit: uint16
                  channels=1,
                  save=True,
                  conf=0.01,
                  # visualize=True # visualize model features maps
                )