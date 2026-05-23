"""
End-to-end pipeline: 画像 → ML 推論 → ベクトル化 → 3D 生成

Streamlit から呼び出すための統合関数。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vectorize import (
    extract_room_polygons,
    extract_wall_segments,
    find_nearest_rooms,
)
from scripts.build_3d_model import build_scene_from_floorplan, export_scene_to_glb_bytes


SEG_CLASS_NAMES = ["door", "room", "shower", "sink", "staircase", "toilet", "wall", "window"]

DEVICE_CONFIG = {
    "door":      {"max_rooms": 2, "max_dist": 80.0},
    "window":    {"max_rooms": 1, "max_dist": 80.0},
    "shower":    {"max_rooms": 1, "max_dist": 50.0},
    "sink":      {"max_rooms": 1, "max_dist": 50.0},
    "staircase": {"max_rooms": 1, "max_dist": 80.0},
    "toilet":    {"max_rooms": 1, "max_dist": 50.0},
}


def run_full_pipeline(
    image_bgr: np.ndarray,
    seg_model_path: str = "models/m1_seg_yolov8n/weights/best.pt",
    conf_threshold: float = 0.25,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """1枚の画像から、JSON + GLB + 中間結果を生成
    
    Args:
        image_bgr: BGR 画像 (OpenCV 形式)
        seg_model_path: M1 セグメンテーションモデルのパス
        conf_threshold: 推論の信頼度しきい値
        progress_callback: (message, percent) を受け取るコールバック
    Returns:
        辞書: {
            "floorplan_json": dict,
            "glb_bytes": bytes,
            "annotated_image": np.ndarray (BGR, 検出結果可視化),
            "stats_3d": dict,
            "timing": dict,
        }
    """
    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)
    
    timing = {}
    h, w = image_bgr.shape[:2]
    
    # 1. モデルロード
    _progress("🔄 モデルロード中...", 0.05)
    t0 = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(seg_model_path)
    timing["model_load_ms"] = (time.time() - t0) * 1000
    
    # 2. 推論
    _progress("🧠 セグメンテーション推論中...", 0.20)
    t0 = time.time()
    results = model.predict(
        source=image_bgr,
        imgsz=1024,
        conf=conf_threshold,
        device=device,
        verbose=False,
    )
    result = results[0]
    timing["inference_ms"] = (time.time() - t0) * 1000
    
    if result.masks is None or len(result.masks) == 0:
        raise RuntimeError("マスクが取得できませんでした。画像が間取り図でない可能性があります。")
    
    # 3. マスク集約
    _progress("🧱 マスクをベクトル化中...", 0.50)
    t0 = time.time()
    room_mask = np.zeros((h, w), dtype=np.uint8)
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    devices_raw = []
    
    for j, mask in enumerate(result.masks.data):
        cls_id = int(result.boxes.cls[j])
        cls_name = SEG_CLASS_NAMES[cls_id]
        conf = float(result.boxes.conf[j])
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        
        if cls_name == "room":
            room_mask = np.maximum(room_mask, mask_resized)
        elif cls_name == "wall":
            wall_mask = np.maximum(wall_mask, mask_resized)
        elif cls_name in DEVICE_CONFIG:
            x1, y1, x2, y2 = result.boxes.xyxy[j].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            devices_raw.append({
                "class": cls_name,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4),
                "center": [cx, cy],
                "room_ids": [],
            })
    
    # 4. ベクトル化
    rooms = extract_room_polygons(room_mask)
    walls = extract_wall_segments(wall_mask)
    
    # 5. 設備 → 部屋紐付け
    for dev in devices_raw:
        config = DEVICE_CONFIG[dev["class"]]
        nearest = find_nearest_rooms(
            tuple(dev["center"]), rooms,
            max_distance=config["max_dist"],
            n=config["max_rooms"],
        )
        dev["room_ids"] = nearest
    
    timing["vectorize_ms"] = (time.time() - t0) * 1000
    
    # 6. JSON 構築
    devices_by_class = {}
    for dev in devices_raw:
        devices_by_class[dev["class"]] = devices_by_class.get(dev["class"], 0) + 1
    
    rooms_json = []
    for room in rooms:
        room_dict = room.to_dict()
        room_dict["devices_in_room"] = [
            {"class": d["class"], "bbox": d["bbox"], "confidence": d["confidence"]}
            for d in devices_raw if room.room_id in d["room_ids"]
        ]
        rooms_json.append(room_dict)
    
    floorplan_data = {
        "image": {"width": w, "height": h},
        "model": "m1_seg_yolov8n",
        "summary": {
            "n_rooms": len(rooms),
            "n_walls": len(walls),
            "n_devices": len(devices_raw),
            "devices_by_class": devices_by_class,
        },
        "rooms": rooms_json,
        "walls": [w.to_dict() for w in walls],
        "devices": devices_raw,
    }
    
    # 7. 3D 生成
    _progress("🏠 3D メッシュを構築中...", 0.80)
    t0 = time.time()
    scene, stats_3d = build_scene_from_floorplan(floorplan_data)
    glb_bytes = export_scene_to_glb_bytes(scene)
    timing["build_3d_ms"] = (time.time() - t0) * 1000
    
    # 8. 検出結果の可視化(2D)
    annotated = result.plot()
    
    _progress("✅ 完了", 1.0)
    
    return {
        "floorplan_json": floorplan_data,
        "glb_bytes": glb_bytes,
        "annotated_image": annotated,
        "stats_3d": stats_3d,
        "timing": timing,
    }
