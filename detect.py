import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    '''
        和原始YOLOv11的detect相比，多了两个参数
            use_simotm="RGB",
            channels=3,
            
        Compared with the original "detect" in YOLOv11, it now has two additional parameters: 
            "use_simotm" set to "RGB". channels=3,
    '''
    model = YOLO(r"runs/M3FD/M3FD_IF-yolo11n2/weights/best.pt") # select your model.pt path
    model.predict(source=r'G:\wan\data\RGBT\M3FD_Detection\images_coco\infrared\trainval',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  show=False,
                  save_frames=False,
                  use_simotm="RGB",
                  channels=3,
                  save=False,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )