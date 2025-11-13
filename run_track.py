# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights as Weights
)
from boxmot.trackers.strongsort.strongsort import StrongSort

# -------------------- 路径配置 --------------------
ROOT_DIR = Path(__file__).resolve().parent
VIDEO_PATH  = ROOT_DIR / "datasets" / "run_track" / "test.mp4"
OUTPUT_PATH = ROOT_DIR / "outputs" / "test_tracked.mp4"
REID_WEIGHTS = ROOT_DIR / "osnet_x0_25_msmt17.pt"

DET_SCORE_THR = 0.3
SUPPRESS_DETS_WHEN_TRACKED = False
ONLY_PERSON = False  # ← 若只想跟踪“人”，改 True

# -------------------- 环境清理 --------------------
if os.environ.get("CUDA_VISIBLE_DEVICES", "").lower() == "cuda":
    os.environ.pop("CUDA_VISIBLE_DEVICES")

# -------------------- 设备选择 --------------------
gpu_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
tracker_device_str = '0' if gpu_ok else 'cpu'
detector_device = torch.device('cuda:0' if gpu_ok else 'cpu')

# -------------------- 检测器 --------------------
weights = Weights.DEFAULT
detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=DET_SCORE_THR)
detector.to(detector_device).eval()
transform = weights.transforms()

# -------------------- StrongSort 跟踪器 --------------------
tracker = StrongSort(
    reid_weights=REID_WEIGHTS,     # 传 Path 最稳
    device=tracker_device_str,     # '0' 或 'cpu'
    half=gpu_ok,
    min_hits=1, n_init=1, min_conf=0.1,
)

# -------------------- COCO 91 -> 80 映射 & 类名 --------------------
# -1 表示该 id 不在 COCO-80 中（会被过滤）
coco91to80 = np.full(91, -1, dtype=int)
_pairs = {
    1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 13:11, 14:12, 15:13,
    16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21, 24:22, 25:23, 27:24, 28:25,
    31:26, 32:27, 33:28, 34:29, 35:30, 36:31, 37:32, 38:33, 39:34, 40:35, 41:36, 42:37,
    43:38, 44:39, 46:40, 47:41, 48:42, 49:43, 50:44, 51:45, 52:46, 53:47, 54:48, 55:49,
    56:50, 57:51, 58:52, 59:53, 60:54, 61:55, 62:56, 63:57, 64:58, 65:59, 67:60, 70:61,
    72:62, 73:63, 74:64, 75:65, 76:66, 77:67, 78:68, 79:69, 80:70, 81:71, 82:72, 84:73,
    85:74, 86:75, 87:76, 88:77, 89:78, 90:79
}
for k, v in _pairs.items():
    coco91to80[k] = v

# COCO-80 类名（0~79），person=0
COCO80_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog",
    "horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
    "bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

def id2color(tid: int):
    r = (tid * 37) % 255; g = (tid * 17) % 255; b = (tid * 29) % 255
    return int(b), int(g), int(r)

def draw_label(frame, x1, y1, tid, cls, conf, color):
    name = COCO80_NAMES[int(cls)] if 0 <= int(cls) < len(COCO80_NAMES) else f"c{int(cls)}"
    label = f"ID {tid} | {name} | {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y1_txt = max(0, y1 - th - 6)
    cv2.rectangle(frame, (x1, y1_txt), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

def draw_detections(frame, boxes, scores, labels, thr=0.3, mask=None):
    if mask is None:
        mask = scores >= thr
    for (x1, y1, x2, y2), sc, lb in zip(boxes[mask], scores[mask], labels[mask]):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180,180,180), 2)
        name = COCO80_NAMES[int(lb)] if 0 <= int(lb) < len(COCO80_NAMES) else f"c{int(lb)}"
        cv2.putText(frame, f"det {name} {float(sc):.2f}",
                    (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200,200,200), 1, cv2.LINE_AA)

def iou_matrix(a, b):
    if len(a)==0 or len(b)==0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
    inter_y1 = np.maximum(ay1[:,None], by1[None,:])
    inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
    inter_y2 = np.minimum(ay2[:,None], by2[None,:])
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h
    area_a = (ax2-ax1) * (ay2-ay1)
    area_b = (bx2-bx1) * (by2-by1)
    union = area_a[:,None] + area_b[None,:] - inter
    return np.where(union > 0, inter/union, 0.0)

# -------------------- 打开视频 --------------------
if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"未找到视频：{VIDEO_PATH}")

cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (W, H))
if not writer.isOpened():
    writer = None
    print("[WARN] 无法创建输出视频，将仅显示。")

print(f"读取视频：{VIDEO_PATH}\n输出结果：{OUTPUT_PATH}\n按 q 退出。")
frame_idx = 0

with torch.inference_mode():
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # --- 检测 ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inp = transform(pil_img).to(detector_device)
        out = detector([inp])[0]

        scores_91 = out['scores'].detach().cpu().numpy()
        boxes_91  = out['boxes'].detach().cpu().numpy()
        labels_91 = out['labels'].detach().cpu().numpy()  # 这是 COCO-91 索引

        # 过滤置信度
        keep_sc = scores_91 >= DET_SCORE_THR
        boxes_91, labels_91, scores_91 = boxes_91[keep_sc], labels_91[keep_sc], scores_91[keep_sc]

        # 91 -> 80
        labels_80 = coco91to80[labels_91]         # [-1 或 0..79]
        keep_map = labels_80 != -1
        boxes, labels, scores = boxes_91[keep_map], labels_80[keep_map], scores_91[keep_map]

        # 只保留“人”（COCO-80 的 person=0）
        if ONLY_PERSON:
            kp = labels == 0
            boxes, labels, scores = boxes[kp], labels[kp], scores[kp]

        # 拼 dets: N×6 [x1,y1,x2,y2,conf,cls]
        if boxes.shape[0] > 0:
            dets = np.concatenate([boxes, scores[:, None], labels[:, None]], axis=1).astype(np.float32)
        else:
            dets = np.zeros((0, 6), dtype=np.float32)

        # --- 跟踪 ---
        tracks = tracker.update(dets, frame)  # 输出: [x1,y1,x2,y2,id,conf,cls,ind]

        # 1) 灰色检测框（可选抑制与轨迹重叠的检测）
        det_mask = np.ones(len(scores), dtype=bool)
        if SUPPRESS_DETS_WHEN_TRACKED and tracks is not None and len(tracks) > 0:
            iou = iou_matrix(boxes, tracks[:, 0:4])
            covered = (iou.max(axis=1) > 0.5)
            det_mask &= ~covered
        draw_detections(frame, boxes, scores, labels, thr=DET_SCORE_THR, mask=det_mask)

        # 2) 轨迹（带历史）
        tracker.plot_results(frame, show_trajectories=True)

        # 3) 彩色标签（80类名）
        if tracks is not None and len(tracks) > 0:
            for t in tracks:
                x1, y1, x2, y2, tid, tconf, tcls, _ = t
                x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
                color = id2color(tid)
                draw_label(frame, x1, y1, tid, tcls, float(tconf), color)

        cv2.putText(frame, f"{frame_idx:06d} Grey=Det  Color=Tracks",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2, cv2.LINE_AA)

        if writer: writer.write(frame)
        cv2.imshow("StrongSort Video Tracking (COCO-80 labels)", frame)
        if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord('q'):
            break

cap.release()
if writer: writer.release()
cv2.destroyAllWindows()
