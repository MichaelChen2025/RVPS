import time
import cv2
import torch
import numpy as np
import sys
import os
from ultralytics import YOLO

# =========================
# 配置
# =========================

# 默认使用 OBS 虚拟摄像头 / 摄像头
SOURCE = 0

# 如果要使用本地视频，取消下面这一行注释，并修改路径
# SOURCE = r"G:\Videos\road_test.mp4"
# SOURCE = r"./videos/test.mp4"

DISPLAY_SIZE = (1280, 720)

# YOLOP 输入尽量用 32 的倍数
YOLO_SIZE = (960, 544)
SEG_SIZE = (960, 544)

SEG_EVERY_N_FRAMES = 2
YOLO_CONF = 0.35

KEEP_CLASSES = {"person", "car", "bus", "truck", "traffic light", "stop sign"}

WINDOW_NAME = "RVPS - Road Vision Perception System"

# YOLOP 本地目录
YOLOP_DIR = r"G:\PythonProject\DeepLabTest\YOLOP"
YOLOP_WEIGHT = os.path.join(YOLOP_DIR, "weights", "End-to-end.pth")

# =========================
# 设备
# =========================
if torch.cuda.is_available():
    DEVICE = "cuda:0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("Using device:", DEVICE)

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# =========================
# 加载 YOLOP 本地模块
# =========================
if YOLOP_DIR not in sys.path:
    sys.path.append(YOLOP_DIR)

from lib.models import get_net
from lib.config import cfg

# =========================
# 模型
# =========================
yolo_model = YOLO("yolov8n.pt")
yolo_model.to(DEVICE)

if not os.path.exists(YOLOP_WEIGHT):
    raise FileNotFoundError(f"未找到 YOLOP 权重文件: {YOLOP_WEIGHT}")

yolop_model = get_net(cfg)
checkpoint = torch.load(YOLOP_WEIGHT, map_location=DEVICE)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    yolop_model.load_state_dict(checkpoint["state_dict"])
else:
    yolop_model.load_state_dict(checkpoint)

yolop_model = yolop_model.to(DEVICE)
yolop_model.eval()

# =========================
# 工具函数
# =========================
def preprocess_yolo(frame):
    return cv2.resize(frame, YOLO_SIZE, interpolation=cv2.INTER_LINEAR)


def preprocess_seg(frame):
    img = cv2.resize(frame, SEG_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    t = torch.from_numpy(rgb).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    return t


def restore_mask(mask, shape):
    return cv2.resize(
        mask.astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST
    )


def clean_mask(mask, kernel_size=3):
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def scale_boxes_to_original(boxes_xyxy, infer_shape, original_shape):
    infer_h, infer_w = infer_shape[:2]
    orig_h, orig_w = original_shape[:2]

    scale_x = orig_w / infer_w
    scale_y = orig_h / infer_h

    scaled = []

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        scaled.append([
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        ])

    return scaled


# =========================
# 可行驶区域：青色叠加
# =========================
def draw_drivable(frame, da_mask):
    out = frame.copy()
    h = da_mask.shape[0]

    da_mask = da_mask.copy()
    da_mask[:h // 2, :] = 0
    da_mask = clean_mask(da_mask, 5)

    overlay = np.zeros_like(frame)
    overlay[da_mask > 0] = (255, 255, 0)

    out = cv2.addWeighted(out, 0.86, overlay, 0.14, 0)

    contours, _ = cv2.findContours(
        da_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(out, contours, -1, (255, 255, 0), 2)

    return out


# =========================
# 车道线：红色点
# =========================
def draw_lane(frame, ll_mask):
    out = frame.copy()

    ll_mask = ll_mask.copy()
    h = ll_mask.shape[0]

    ll_mask[:h // 2, :] = 0
    ll_mask = clean_mask(ll_mask, 3)

    ys, xs = np.where(ll_mask > 0)

    if len(xs) > 0:
        step = max(1, len(xs) // 7000)

        for x, y in zip(xs[::step], ys[::step]):
            cv2.circle(out, (int(x), int(y)), 1, (0, 0, 255), -1)

    return out


# =========================
# YOLO 检测框
# =========================
def get_box_color(name):
    if name == "person":
        return (0, 165, 255)

    if name in {"car", "bus", "truck"}:
        return (0, 255, 0)

    if name in {"traffic light", "stop sign"}:
        return (0, 255, 255)

    return (255, 255, 255)


def draw_yolo_boxes(frame, result, infer_shape, original_shape):
    out = frame.copy()

    if result is None or result.boxes is None or len(result.boxes) == 0:
        return out

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    scaled_boxes = scale_boxes_to_original(
        boxes,
        infer_shape,
        original_shape
    )

    for box, cls_id, conf in zip(scaled_boxes, classes, confs):
        name = yolo_model.names[int(cls_id)]

        if name not in KEEP_CLASSES:
            continue

        x1, y1, x2, y2 = box
        color = get_box_color(name)
        label = f"{name} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            out,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    return out


# =========================
# 视频 / 摄像头输入
# =========================

# Windows 摄像头 / OBS 虚拟摄像头建议使用 CAP_DSHOW
if isinstance(SOURCE, int):
    cap = cv2.VideoCapture(SOURCE, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(SOURCE)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(f"无法打开输入源: {SOURCE}")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# =========================
# 缓存
# =========================
last_da = None
last_ll = None
last_yolo_result = None
last_yolo_infer_shape = None

frame_id = 0
prev_time = time.time()

# =========================
# 主循环
# =========================
with torch.no_grad():
    while True:
        # 摄像头模式下丢弃旧帧，降低延迟
        if isinstance(SOURCE, int):
            cap.grab()
            cap.grab()

        ret, frame = cap.read()

        if not ret or frame is None:
            print("视频结束或读取失败")
            break

        frame = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
        original = frame.copy()

        # 1) YOLO 目标检测
        yolo_frame = preprocess_yolo(original)

        yolo_results = yolo_model.predict(
            source=yolo_frame,
            conf=YOLO_CONF,
            verbose=False,
            device=DEVICE
        )

        last_yolo_result = yolo_results[0]
        last_yolo_infer_shape = yolo_frame.shape

        # 2) YOLOP 分割：可行驶区域 + 车道线
        if frame_id % SEG_EVERY_N_FRAMES == 0 or last_da is None or last_ll is None:
            seg_tensor = preprocess_seg(original)
            seg_out = yolop_model(seg_tensor)

            if isinstance(seg_out, (list, tuple)) and len(seg_out) == 3:
                _, da, ll = seg_out
            else:
                raise RuntimeError(f"YOLOP 输出格式异常: {type(seg_out)}")

            da = torch.argmax(da, 1).squeeze().cpu().numpy().astype(np.uint8)
            ll = torch.argmax(ll, 1).squeeze().cpu().numpy().astype(np.uint8)

            last_da = restore_mask(da, original.shape)
            last_ll = restore_mask(ll, original.shape)

        # 3) 绘制结果叠加
        vis = draw_drivable(original, last_da)
        vis = draw_lane(vis, last_ll)
        vis = draw_yolo_boxes(
            vis,
            last_yolo_result,
            last_yolo_infer_shape,
            original.shape
        )

        # 4) 显示 FPS 和状态信息
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(
            vis,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        status = (
            f"Display: {DISPLAY_SIZE[0]}x{DISPLAY_SIZE[1]}  "
            f"YOLO: {YOLO_SIZE[0]}x{YOLO_SIZE[1]}  "
            f"SEG: {SEG_SIZE[0]}x{SEG_SIZE[1]}"
        )

        cv2.putText(
            vis,
            status,
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            vis,
            "Object Detection + Lane Detection + Drivable Area Segmentation",
            (20, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(WINDOW_NAME, vis)

        frame_id += 1

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break

cap.release()
cv2.destroyAllWindows()