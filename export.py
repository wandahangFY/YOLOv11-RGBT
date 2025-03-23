import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples

if __name__ == '__main__':
    model = YOLO('runs/M3FD/M3FD-yolo11n-RGBT-midfusion-RGBRGB6C-e300-16-/weights/best.pt')
    model.export(format='onnx', simplify=True, opset=13,channels=6)