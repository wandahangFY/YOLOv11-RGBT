import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'LLVIP/LLVIP-yolov8-RGBT-midfusion/weights/best.pt')
    model.val(data=r'ultralytics/cfg/datasets/LLVIP.yaml',
              split='val',
              imgsz=640,
              batch=16,
              use_simotm="RGBT",
              channels=4,
              # pairs_rgb_ir=['visible','infrared'] , # default: ['visible','infrared'] , others: ['rgb', 'ir'],  ['images', 'images_ir'], ['images', 'image']
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val/LLVIP',
              name='LLVIP_r20-yolov8n-no_pretrained',
              )