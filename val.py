import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'runs/LLVIP/LLVIP-yolo11n-RGBRGB6C-earlyfusion3/weights/best.pt')
    model.val(data=r'ultralytics/cfg/datasets/LLVIP.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # use_simotm="RGBT",  # 4通道 RGB + IR
              # channels=4,

              use_simotm="RGBRGB6C", # 6 通道 RGB + IR(3通道)
              channels=6,

              # use_simotm="RGB",  # 3 通道 RGB 其余模式类似
              # channels=3,

              # pairs_rgb_ir=['visible','infrared'] , # default: ['visible','infrared'] , others: ['rgb', 'ir'],  ['images', 'images_ir'], ['images', 'image']
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val/LLVIP',
              name='LLVIP_r20-yolov8n-no_pretrained',
              )