import warnings

import torch

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
#     model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-RDBB.yaml')
#     # model.load('yolov8n.pt') # loading pretrain weights
#     model.train(data=R'ultralytics/cfg/datasets/KAIST.yaml',
#                 cache=False,
#                 imgsz=640,
#                 epochs=50,
#                 batch=16,
#                 close_mosaic=0,
#                 workers=2,
#                 device='0',
#                 optimizer='SGD',  # using SGD
#                 # resume='', # last.pt path
#                 # amp=False, # close amp
#                 # fraction=0.2,
#                 use_simotm="RGBT",
#                 channels=4,
#                 project='runs/KAIST',
#                 name='KAIST-yolo11-RGBT-earlyfusion-e50-16-',
#                 )


    model = YOLO('ultralytics/cfg/models/v5/yolov5n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/KAIST_IF.yaml',
                cache=False,
                imgsz=640,
                epochs=50,
                batch=16,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGB",
                channels=3,
                project='runs/KAIST',
                name='KAIST_IF-yolov5n-RGB-e50-16-',
                )
    del model
    torch.cuda.empty_cache()

    model = YOLO('ultralytics/cfg/models/v9/yolov9t.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/KAIST_IF.yaml',
                cache=False,
                imgsz=640,
                epochs=50,
                batch=16,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGB",
                channels=3,
                project='runs/KAIST',
                name='KAIST_IF-yolov9t-RGB-e50-16-',
                )
    del model
    torch.cuda.empty_cache()

    model = YOLO('ultralytics/cfg/models/v10/yolov10n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/KAIST_IF.yaml',
                cache=False,
                imgsz=640,
                epochs=50,
                batch=16,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGB",
                channels=3,
                project='runs/KAIST',
                name='KAIST_IF-yolov10n-RGB-e50-16-',
                )
    del model
    torch.cuda.empty_cache()

    model = YOLO('ultralytics/cfg/models/v6/yolov6n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/KAIST_IF.yaml',
                cache=False,
                imgsz=640,
                epochs=50,
                batch=16,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGB",
                channels=3,
                project='runs/KAIST',
                name='KAIST_IF-yolov6n-RGB-e50-16-',
                )
    del model
    torch.cuda.empty_cache()

    model = YOLO('ultralytics/cfg/models/v10-RGBT/yolov10n-RGBT-midfusion.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/KAIST.yaml',
                cache=False,
                imgsz=640,
                epochs=50,
                batch=16,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBT",
                channels=4,
                project='runs/KAIST',
                name='KAIST-yolov10n-RGBT-midfusion-e50-16-',
                )
    del model
    torch.cuda.empty_cache()

    model = YOLO('ultralytics/cfg/models/v9-RGBT/yolov9t-RGBT-midfusion.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/KAIST.yaml',
                cache=False,
                imgsz=640,
                epochs=50,
                batch=16,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBT",
                channels=4,
                project='runs/KAIST',
                name='KAIST-yolov9t-RGBT-midfusion-e50-16-',
                )
    del model
    torch.cuda.empty_cache()