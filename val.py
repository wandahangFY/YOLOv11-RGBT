import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'LLVIP_r20/LLVIP_r20-yolov8-RGBRGB6C-midfusion8/weights/best.pt')
    model.val(data=r'ultralytics/cfg/datasets/LLVIP_r20.yaml',
              split='val',
              imgsz=640,
              batch=16,
              use_simotm="RGBT",
              channels=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val/LLVIP_r20',
              name='LLVIP_r20-yolov8n-no_pretrained',
              )