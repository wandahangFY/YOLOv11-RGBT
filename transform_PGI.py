import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics.nn.tasks import DetectionModel

# 辅助头部分的代码从魔鬼面具v11代码里面复制，已和魔导沟通确认，具体请参考  https://github.com/z1069614715/objectdetection_script
if __name__ == '__main__':
    model_PGI_weights_path = 'runs/train/yolo11n-PGI/weights/best.pt'
    model_cfg_path = "ultralytics/cfg/models/11/yolo11n.yaml"
    layer_num, pgi_layer_num = 23, 39
    
    device = torch.device("cpu")
    model_PGI = torch.load(model_PGI_weights_path, map_location='cpu')
    model_name_key = 'model' if model_PGI['model'] is not None else 'ema'
    model_PGI_dict = model_PGI[model_name_key].model.state_dict()
    model_PGI_head = model_PGI[model_name_key].model[-1]
    model_name = model_PGI[model_name_key].names
    model = DetectionModel(model_cfg_path, nc=model_PGI_head.nc)
    model.names = model_name
    model_dict = model.state_dict()
    
    new_dict = {}
    for name in model_PGI_dict:
        layer_id = int(name.split('.')[0]) - 1
        new_name = f'.'.join(['model', str(layer_id)] + name.split('.')[1:])
        if new_name in model_dict and model_PGI_dict[name].size() == model_dict[new_name].size():
            new_dict[new_name] = model_PGI_dict[name]
    
        if (layer_id + 1) == pgi_layer_num:
            new_name = f'.'.join(['model', str(layer_num)] + name.split('.')[1:])
            if new_name in model_dict and model_PGI_dict[name].size() == model_dict[new_name].size():
                new_dict[new_name] = model_PGI_dict[name]
    
    print(len(new_dict), len(model_dict))
    model.load_state_dict(new_dict)
    model.eval()
    torch.save({'model':model.half()}, f'{model_PGI_weights_path[:model_PGI_weights_path.rfind(".")]}_rep.pt')