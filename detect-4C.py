import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # image
    '''
        source 需要和train/val目录一致，且需要包含visible字段，visible同级目录下存在infrared目录，原理是将visible替换为infrared，加载双光谱数据
        
        The source needs to be in the same directory as the train/val directories, and it must contain the "visible" field. 
        There is an "infrared" directory at the same level as the "visible" directory. 
        The principle is to replace "visible" with "infrared" and load the dual-spectrum data.
    '''
    model = YOLO(r"runs/LLVIP/LLVIP-yolov5-RGBT-midfusion/weights/best.pt") # select your model.pt path
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

    # OBB task

    # model = YOLO(r"runs/dota8/dota8-yolo11n-RGBT-midfusion-obb-e300-16-21/weights/best.pt")  # select your model.pt path
    # model.predict(source=r'G:\wan\data\dota8\dota8\images\train',
    #               imgsz=640,
    #               project='runs/detect',
    #               name='exp',
    #               show=False,
    #               task='obb',
    #               save_frames=True,
    #               use_simotm="RGBT",
    #               channels=4,
    #               save=True,
    #               #conf=0.0005,
    #               # visualize=True # visualize model features maps
    #               )

    # # VIDEO
    # model = YOLO(R"runs/M3FD/M3FD-yolov5-RGBT-midfusion/weights/best.pt") # select your model.pt path
    # model.predict(source=r"G:\wan\data\RGBT\testVD\visible\video.mp4",
    #               imgsz=640,
    #               project='runs/detect',
    #               name='exp',
    #               show=False,
    #               save_frames=True,
    #               use_simotm="RGBT",
    #               channels=4,
    #               save=True,
    #               # conf=0.2,
    #               # visualize=True # visualize model features maps
    #             )