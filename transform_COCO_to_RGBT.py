import torch
import torch.nn as nn
import re
import datetime


def copy_and_modify_layers(source_model_path, target_model_path, output_model_path,
                           copy_ranges=[
                               # (源层范围, 目标层范围)
                               ((0, 8), (2, 10)),  # model.0-8 复制到 model.2-10
                               ((0, 8), (12, 20)),  # model.0-8 复制到 model.12-20
                               ((9, 22), (24, 37)),  # model.9-22 复制到 model.24-37
                           ]
                           ):
    # 加载源模型和目标模型
    source_model = torch.load(source_model_path)
    # 如果模型是一个字典，尝试提取模型部分
    if isinstance(source_model, dict) and 'model' in source_model:
        source_model = source_model['model']

    target_model = torch.load(target_model_path)
    # 如果模型是一个字典，尝试提取模型部分
    if isinstance(target_model, dict) and 'model' in target_model:
        target_model = target_model['model']



    # 遍历每个复制范围
    for (src_start, src_end), (tgt_start, tgt_end) in copy_ranges:
        # 获取源模型的层
        source_layers = []
        for name, module in source_model.named_modules():
            if name.startswith('model.') and re.match(r'model\.\d+', name):
                match = re.match(r'model\.(\d+)', name)
                if match:
                    layer_index = int(match.group(1))
                    if src_start <= layer_index <= src_end:
                        source_layers.append((name, module))

        # 获取目标模型的层
        target_layers = []
        for name, module in target_model.named_modules():
            if name.startswith('model.') and re.match(r'model\.\d+', name):
                match = re.match(r'model\.(\d+)', name)
                if match:
                    layer_index = int(match.group(1))
                    if tgt_start <= layer_index <= tgt_end:
                        target_layers.append((name, module))

        # 确保两者的层数量一致
        # assert len(source_layers) == len(target_layers), f"层数量不一致: 源层 {len(source_layers)}, 目标层 {len(target_layers)}"

        # 复制权重
        for (source_name, source_module), (target_name, target_module) in zip(source_layers, target_layers):
            try:
                if isinstance(source_module, nn.Module) and isinstance(target_module, nn.Module):
                    # 检查是否为 Conv2d 层
                    if isinstance(source_module, nn.Conv2d) and isinstance(target_module, nn.Conv2d):
                        # 检查通道数
                        if target_module.in_channels == 1 and source_module.in_channels == 3:
                            # 如果目标层的输入通道数为1，源层输入通道数为3，则取平均
                            new_weight = source_module.weight.mean(dim=1, keepdim=True)
                            target_module.weight = nn.Parameter(new_weight)
                            print(f"复制并修改权重(Copy and change the weights): {source_name} -> {target_name}")
                        elif target_module.out_channels > source_module.out_channels:
                            # 如果目标层的输出通道数大于源层，则复制源层的通道
                            new_weight = torch.cat([source_module.weight] * (target_module.out_channels // source_module.out_channels), dim=0)
                            target_module.weight = nn.Parameter(new_weight)
                            print(f"复制并增加通道数(Copy and increase the number of channels): {source_name} -> {target_name}")
                        else:
                            # 直接复制权重
                            target_module.load_state_dict(source_module.state_dict())
                            print(f"复制权重(copy weights): {source_name} -> {target_name}")
                    else:
                        # 直接复制其他类型的模块
                        target_module.load_state_dict(source_module.state_dict())
                        print(f"复制权重(copy weights): {source_name} -> {target_name}")
            except:
                continue



    # 创建元数据字典
    metadata = {
        'date': datetime.datetime.now().isoformat(),
        'version': '8.3.75',
        'license': 'AGPL-3.0 License (https://ultralytics.com/license)',
        'docs': 'https://docs.ultralytics.com',
        'epoch': -1,
        'best_fitness': None,
        'model': target_model
    }

    # 保存模型和元数据
    torch.save(metadata, output_model_path)
    print(f"最终模型已成功保存到(save to): {output_model_path}")


# 使用示例
copy_and_modify_layers(
    source_model_path='yolo11n.pt', # COCO 预训练模型
    target_model_path=r"./runs/M3FD-yolo11n-RGBT-midfusion-e300-16-/weights/best.pt", # 提前训练一个模型 2 epoch 得到初始结构，用于替换
    output_model_path='yolo11n-RGBT-midfussion.pt', # 最终RGBT 的预训练模型

    copy_ranges = [
        # (源层范围, 目标层范围)
        ((0, 8), (2, 10)),  # model.0-8 复制到 model.2-10
        ((0, 8), (12, 20)),  # model.0-8 复制到 model.12-20
        ((9, 23), (24, 38)),  # model.9-23 复制到 model.24-38
    ],
)