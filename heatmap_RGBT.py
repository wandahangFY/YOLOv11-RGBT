# 代码来源于B站魔鬼面具  https://github.com/z1069614715/objectdetection_script

import math
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys, copy
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, \
    AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


# from
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    value = (114, 114, 114)  # RGB 彩色图像
    # 根据通道数设置 value 的维度
    print(im.shape)
    channels = 3
    if len(im.shape) > 2:
        channels = im.shape[2]
    else:
        channels = 1
    if channels == 1:
        value = (114, 114, 114)  # 单通道灰度图像
    elif channels == 3 or channels == 6:
        value = (114, 114, 114)  # RGB 彩色图像
    elif channels == 4:
        value = (114, 114, 114, 114)  # RGB 彩色图像
    else:
        value = (114, 114, 114)  # RGB 彩色图像
        pass
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)  # add border
    return im, ratio, (top, bottom, left, right)


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        if self.model.end2end:
            logits_ = result[:, :, 4:]
            boxes_ = result[:, :, :4]
            sorted, indices = torch.sort(logits_[:, :, 0], descending=True)
            return logits_[0][indices[0]], boxes_[0][indices[0]]
        elif self.model.task == 'detect':
            logits_ = result[:, 4:]
            boxes_ = result[:, :4]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
                indices[0]]
        elif self.model.task == 'segment':
            logits_ = result[0][:, 4:4 + self.model.nc]
            boxes_ = result[0][:, :4]
            mask_p, mask_nm = result[1][2].squeeze(), result[1][1].squeeze().transpose(1, 0)
            c, h, w = mask_p.size()
            mask = (mask_nm @ mask_p.view(c, -1))
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
                indices[0]], mask[indices[0]]
        elif self.model.task == 'pose':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            poses_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
                indices[0]], torch.transpose(poses_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'obb':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            angles_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
                indices[0]], torch.transpose(angles_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'classify':
            return result[0]

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        if self.model.task == 'detect':
            post_result, pre_post_boxes = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes]]
        elif self.model.task == 'segment':
            post_result, pre_post_boxes, pre_post_mask = self.post_process(model_output)
            return [[post_result, pre_post_boxes, pre_post_mask]]
        elif self.model.task == 'pose':
            post_result, pre_post_boxes, pre_post_pose = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_pose]]
        elif self.model.task == 'obb':
            post_result, pre_post_boxes, pre_post_angle = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_angle]]
        elif self.model.task == 'classify':
            data = self.post_process(model_output)
            return [data]

    def release(self):
        for handle in self.handles:
            handle.remove()


class yolo_detect_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio, end2end) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
        self.end2end = end2end

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if (self.end2end and float(post_result[i, 0]) < self.conf) or (
                    not self.end2end and float(post_result[i].max()) < self.conf):
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                if self.end2end:
                    result.append(post_result[i, 0])
                else:
                    result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)


class yolo_segment_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        post_result, pre_post_boxes, pre_post_mask = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'segment' or self.ouput_type == 'all':
                result.append(pre_post_mask[i].mean())
        return sum(result)


class yolo_pose_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        post_result, pre_post_boxes, pre_post_pose = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'pose' or self.ouput_type == 'all':
                result.append(pre_post_pose[i].mean())
        return sum(result)


class yolo_obb_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        post_result, pre_post_boxes, pre_post_angle = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'obb' or self.ouput_type == 'all':
                result.append(pre_post_angle[i])
        return sum(result)


class yolo_classify_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        return data.max()


class yolo_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_result, renormalize,
                 task, img_size, use_simotm, channels):
        device = torch.device(device)
        model_yolo = YOLO(weight)
        model_names = model_yolo.names
        print(f'model class info:{model_names}')
        model = copy.deepcopy(model_yolo.model)
        model.to(device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        self.use_simotm = 'RGBT'
        self.channels = 4
        model.task = task
        if not hasattr(model, 'end2end'):
            model.end2end = False

        if task == 'detect':
            target = yolo_detect_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'segment':
            target = yolo_segment_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'pose':
            target = yolo_pose_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'obb':
            target = yolo_obb_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'classify':
            target = yolo_classify_target(backward_type, conf_threshold, ratio, model.end2end)
        else:
            raise Exception(f"not support task({task}).")

        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)  # 绘制检测框
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)  # 绘制类别、置信度
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1]
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path, save_path):
        # img process
        f = img_path
        print("self.use_simotm=", self.use_simotm)
        try:
            # img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)

            if self.use_simotm == 'Gray2BGR':
                im = cv2.imread(f)  # BGR
            elif self.use_simotm == 'Gray':
                im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # GRAY
            elif self.use_simotm == 'Gray16bit':
                im = cv2.imread(f, cv2.IMREAD_UNCHANGED)  # GRAY
                im = im.astype(np.float32)

            elif self.use_simotm == 'RGBT':
                im_visible = cv2.imread(f)  # BGR
                im_infrared = cv2.imread(f.replace('visible', 'infrared'), cv2.IMREAD_GRAYSCALE)  # BGR
                if im_visible is None or im_infrared is None:
                    raise FileNotFoundError(f"Image Not Found {f}")
                h_vis, w_vis = im_visible.shape[:2]  # orig hw
                h_inf, w_inf = im_infrared.shape[:2]  # orig hw

                if h_vis != h_inf or w_vis != w_inf:

                    r_vis = self.imgsz / max(h_vis, w_vis)  # ratio
                    r_inf = self.imgsz / max(h_inf, w_inf)  # ratio
                    if r_vis != 1:  # if sizes are not equal
                        interp = cv2.INTER_LINEAR if (self.augment or r_vis > 1) else cv2.INTER_AREA
                        im_visible = cv2.resize(im_visible, (
                            min(math.ceil(w_vis * r_vis), self.imgsz), min(math.ceil(h_vis * r_vis), self.imgsz)),
                                                interpolation=interp)
                    if r_inf != 1:  # if sizes are not equal
                        interp = cv2.INTER_LINEAR if (self.augment or r_inf > 1) else cv2.INTER_AREA
                        im_infrared = cv2.resize(im_infrared, (
                            min(math.ceil(w_inf * r_inf), self.imgsz), min(math.ceil(h_inf * r_inf), self.imgsz)),
                                                 interpolation=interp)

                # 将彩色图像的三个通道分离
                b, g, r = cv2.split(im_visible)
                # 合并成四通道图像
                im = cv2.merge((b, g, r, im_infrared))
            elif self.use_simotm == 'RGBRGB6C':
                im_visible = cv2.imread(f)  # BGR
                im_infrared = cv2.imread(f.replace('visible', 'infrared'))  # BGR
                if im_visible is None or im_infrared is None:
                    raise FileNotFoundError(f"Image Not Found {f}")
                h_vis, w_vis = im_visible.shape[:2]  # orig hw
                h_inf, w_inf = im_infrared.shape[:2]  # orig hw

                if h_vis != h_inf or w_vis != w_inf:

                    r_vis = self.imgsz / max(h_vis, w_vis)  # ratio
                    r_inf = self.imgsz / max(h_inf, w_inf)  # ratio
                    if r_vis != 1:  # if sizes are not equal
                        interp = cv2.INTER_LINEAR if (self.augment or r_vis > 1) else cv2.INTER_AREA
                        im_visible = cv2.resize(im_visible, (
                            min(math.ceil(w_vis * r_vis), self.imgsz), min(math.ceil(h_vis * r_vis), self.imgsz)),
                                                interpolation=interp)
                    if r_inf != 1:  # if sizes are not equal
                        interp = cv2.INTER_LINEAR if (self.augment or r_inf > 1) else cv2.INTER_AREA
                        im_infrared = cv2.resize(im_infrared, (
                            min(math.ceil(w_inf * r_inf), self.imgsz), min(math.ceil(h_inf * r_inf), self.imgsz)),
                                                 interpolation=interp)

                # 将彩色图像的三个通道分离
                b, g, r = cv2.split(im_visible)
                b2, g2, r2 = cv2.split(im_infrared)
                # 合并成6通道图像
                im = cv2.merge((b, g, r, b2, g2, r2))
            else:
                im = cv2.imread(f)  # BGR
        except:
            raise FileNotFoundError(f"Image Not Found {f}")
        img = im
        src_w,src_h=img.shape[1],img.shape[0]

        img, _, (top, bottom, left, right) = letterbox(img, new_shape=(self.img_size, self.img_size),
                                                       auto=True)  # 如果需要完全固定成宽高一样就把auto设置为False
        dst_w, dst_h = img.shape[1], img.shape[0]

        print(dst_w, dst_h,src_w,src_h)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.float32(img) / 255.0
        # tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        if (img.shape[2] == 1):
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
        elif (img.shape[2] == 4):
            img3c = np.ascontiguousarray(img.transpose(2, 0, 1)[:3, :, :][::-1])
            img1c = img.transpose(2, 0, 1)[-1:, :, :]
            img = np.concatenate((img3c, img1c), axis=0)
            # img = torch.from_numpy(img)
            # ----------------------------   3 _format_img
        elif (img.shape[2] == 6):
            img3c = np.ascontiguousarray(img.transpose(2, 0, 1)[:3, :, :][::-1])
            img3c2 = np.ascontiguousarray(img.transpose(2, 0, 1)[3:, :, :][::-1])
            img = np.concatenate((img3c, img3c2), axis=0)
        else:
            img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1])
        # img = torch.from_numpy(img)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        print(f'tensor size:{tensor.size()}')

        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            print(f"Warning... self.method(tensor, [self.target]) failure.")
            return

        grayscale_cam = grayscale_cam[0, :]
        # cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        cam_image = show_cam_on_image(np.transpose(img[:3, :, :], axes=[1, 2, 0]), grayscale_cam, use_rgb=True)
        print(f'cam_image size:{cam_image.shape}')
        pred = self.model_yolo.predict(img_path, conf=self.conf_threshold, iou=0.7, use_simotm=self.use_simotm,
                                       channels=self.channels)[0]

        resize_results(pred, (src_h, src_w), (dst_h, dst_w))

        print(pred.boxes)
        if self.renormalize and self.task in ['detect', 'segment', 'pose']:
            cam_image = self.renormalize_cam_in_bounding_boxes(pred.boxes.xyxy.cpu().detach().numpy().astype(np.int32),
                                                               np.transpose(img[:3, :, :], axes=[1, 2, 0]),
                                                               grayscale_cam)
        if self.show_result:
            cam_image = pred.plot(img=cam_image,
                                  conf=True,  # 显示置信度
                                  font_size=None,  # 字体大小，None为根据当前image尺寸计算
                                  line_width=None,  # 线条宽度，None为根据当前image尺寸计算
                                  labels=False,  # 显示标签
                                  use_simotm=self.use_simotm,
                                  )

        # 去掉padding边界
        cam_image = cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right]
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)

    def __call__(self, img_path, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
        else:
            self.process(img_path, f'{save_path}/result.png')


def resize_results(results, orig_shape, target_shape):
    """
    将检测结果从原始尺寸缩放到目标尺寸

    Args:
        results: YOLO Results对象
        orig_shape: (height, width) 原始图像尺寸
        target_shape: (height, width) 目标尺寸
    """
    if len(results.boxes) == 0:
        return results

    # 获取框数据并克隆（避免修改原始数据）
    boxes = results.boxes.data.clone()

    # 缩放
    orig_h, orig_w = orig_shape
    target_h, target_w = target_shape

    scale_h = target_h / orig_h
    scale_w = target_w / orig_w

    boxes[:, [0, 2]] *= scale_w
    boxes[:, [1, 3]] *= scale_h

    # 更新结果
    results.update(boxes=boxes)

    # 更新Results对象中的orig_img（如果需要）
    if hasattr(results, 'orig_shape'):
        results.orig_shape = target_shape

    return results


def get_params():
    params = {
        # 'weight': 'LLVIP-yolo11n-RGBT-midfusion-MCF.pt',
        'weight': 'best.pt',
        # 现在只需要指定权重即可,不需要指定cfg M3FD-yolo11n-RGBT-midfusion-MCF.pt  best.pt
        'device': 'cuda:0',
        'method': 'GradCAMPlusPlus',
        # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM
        'layer': [10, 12, 14, 16, 18],  # [10, 12, 14, 16, 18],    [30, 33, 36, 39, 42]
        'backward_type': 'all',
        # detect:<class, box, all> segment:<class, box, segment, all> pose:<box, keypoint, all> obb:<box, angle, all> classify:<all>
        'conf_threshold': 0.2,  # 0.2
        'ratio': 0.02,  # 0.02-0.1
        'show_result': True,  # 不需要绘制结果请设置为False
        'renormalize': True,  # 需要把热力图限制在框内请设置为True(仅对detect,segment,pose有效)
        'task': 'detect',  # 任务(detect,segment,pose,obb,classify)
        'img_size': 640,  # 图像尺寸
        # 'channels': 6,
        # 'use_simotm': 'RGBRGB6C',  # RGBRGB6C    RGB  RGBT
        'channels': 4,
        'use_simotm': 'RGBT',  # RGBRGB6C    RGB  RGBT

    }
    return params


# pip install grad-cam==1.5.4 --no-deps
if __name__ == '__main__':
    model = yolo_heatmap(**get_params())
    model(r"G:\wan\data\RGBT\test\visible", 'result3')
    # model(r'/home/hjj/Desktop/dataset/dataset_coco/coco/images/val2017', 'result')