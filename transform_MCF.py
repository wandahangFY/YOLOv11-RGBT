import torch
import torch.nn as nn
import re
import datetime


def copy_and_modify_layers(source_model_path, target_model_path, output_model_path):
    # 加载源模型和目标模型
    source_model = torch.load(source_model_path)
    # 如果模型是一个字典，尝试提取模型部分
    if isinstance(source_model, dict) and 'model' in source_model:
        source_model = source_model['model']

    target_model = torch.load(target_model_path)
    # 如果模型是一个字典，尝试提取模型部分
    if isinstance(target_model, dict) and 'model' in target_model:
        target_model = target_model['model']

    # 定义复制的层范围
    copy_ranges = [
        # (源层范围, 目标层范围)
        ((0, 4), (2, 6)),  # model.0-8 复制到 model.2-10
        ((0, 0), (10, 10)),  # model.0-8 复制到 model.12-20
        ((1, 4), (11, 14)),  # model.0-8 复制到 model.12-20
        ((5, 6), (17, 18)),  # model.0-8 复制到 model.12-20
        ((5, 6), (19, 20)),  # model.0-8 复制到 model.12-20
        ((7, 8), (23, 24)),  # model.0-8 复制到 model.12-20
        ((7, 8), (25, 26)),  # model.0-8 复制到 model.12-20

        ((9, 23), (29, 43)),  # model.9-22 复制到 model.24-37
    ]

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
                            print(f"复制并修改权重: {source_name} -> {target_name}")
                        elif target_module.out_channels > source_module.out_channels:
                            # 如果目标层的输出通道数大于源层，则复制源层的通道
                            new_weight = torch.cat([source_module.weight] * (target_module.out_channels // source_module.out_channels), dim=0)
                            target_module.weight = nn.Parameter(new_weight)
                            print(f"复制并增加通道数: {source_name} -> {target_name}")
                        else:
                            # 直接复制权重
                            target_module.load_state_dict(source_module.state_dict())
                            print(f"复制权重: {source_name} -> {target_name}")
                    else:
                        # 直接复制其他类型的模块
                        target_module.load_state_dict(source_module.state_dict())
                        print(f"复制权重: {source_name} -> {target_name}")
            except:
                print(f"复制权重失败: {source_name} -> {target_name}")
                continue


    # 定义要设置为0的层名称
    zero_layers = ['model.8','model.15','model.21','model.27']  # 示例：将目标模型的 model.38 层的权重设置为0

    for layer_name in zero_layers:
        # 查找目标层
        target_layer = None
        for name, module in target_model.named_modules():
            if name == layer_name:
                target_layer = module
                break

        # 确保找到目标层
        assert target_layer is not None, f"未找到目标层 {layer_name}"
        # print(type(target_layer))
        # 将目标层的权重设置为0
        if isinstance(target_layer, nn.Conv2d):
            nn.init.zeros_(target_layer.weight)
            if target_layer.bias is not None:
                nn.init.zeros_(target_layer.bias)
            print(f"将层 {layer_name} 的权重设置为0")
            print("2D Conv Weights (sum):", target_layer.weight.sum().item())

    # 创建元数据字典
    metadata = {
        'date': datetime.datetime.now().isoformat(),
        'version': '8.2.100',
        'license': 'AGPL-3.0 License (https://ultralytics.com/license)',
        'docs': 'https://docs.ultralytics.com',
        'epoch': 300,
        'best_fitness': None,
        'model': target_model
    }
    # print(target_model)
    # 保存模型和元数据
    torch.save(metadata, output_model_path)
    print(f"最终模型已成功保存到 {output_model_path}")


# 使用示例           Usage Example
# 如果需要更改模型结构，请仔细阅读本代码，重点修改20行和85行的网络结构层
# If you need to modify the model structure, please carefully read this code and focus on modifying the network structure layers at line 20 and line 85.

copy_and_modify_layers(
    source_model_path=r"E:\Download\RGBT_RESULT\M3FD\M3FD_IF-yolo11n-e300-16-pretrained\weights\best.pt",  # input: step 1
    target_model_path=r"./runs/M3FD/M3FD-yolo11n-RGBT-midfusion-MCF-e300-16-/weights/best.pt", # input: step 2
    output_model_path='M3FD-yolo11n-RGBT-midfusion-MCF.pt'  # output: step 3
)
