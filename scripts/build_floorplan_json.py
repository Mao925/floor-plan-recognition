"""
M1 セグメンテーション結果 + ベクトル化 → 統合 JSON 平面図
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vectorize import (
    extract_room_polygons,
    extract_wall_segments,
    visualize_room_polygons,
    visualize_walls,
    find_nearest_rooms,
)


# ============================================================
# 設定
# ============================================================
MODEL_PATH = "models/m1_seg_yolov8n/weights/best.pt"
TEST_IMAGES_DIR = Path("data/floorplan_seg/test/images")
OUTPUT_DIR = Path("outputs/floorplan_json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["door", "room", "shower", "sink", "staircase", "toilet", "wall", "window"]

# 設備のクラスごとの紐付けルール
# (壁の上にあるか部屋の中にあるか、最大いくつの部屋に紐付くか)
DEVICE_CONFIG = {
    "door":      {"location": "boundary", "max_rooms": 2, "max_dist": 80.0},
    "window":    {"location": "boundary", "max_rooms": 1, "max_dist": 80.0},  # 通常は外壁
    "shower":    {"location": "interior", "max_rooms": 1, "max_dist": 50.0},
    "sink":      {"location": "interior", "max_rooms": 1, "max_dist": 50.0},
    "staircase": {"location": "interior", "max_rooms": 1, "max_dist": 80.0},
    "toilet":    {"location": "interior", "max_rooms": 1, "max_dist": 50.0},
}

DEVICE_COLORS = {
    "door":      (255, 0, 0),
    "window":    (255, 192, 203),
    "shower":    (255, 0, 255),
    "sink":      (0, 165, 255),
    "staircase": (0, 255, 255),
    "toilet":    (128, 0, 128),
}


# ============================================================
# メイン処理
# ============================================================
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用デバイス: {device}")

model = YOLO(MODEL_PATH)
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model.predict(source=dummy, device=device, verbose=False, imgsz=640)
print("✅ モデルロード + warmup 完了\n")

test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))[:3]

for i, img_path in enumerate(test_images, 1):
    print(f"\n{'='*60}")
    print(f"画像 {i}/3: {img_path.name[:50]}")
    print(f"{'='*60}")
    
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    # 推論
    start = time.time()
    results = model.predict(source=img, imgsz=1024, conf=0.25, device=device, verbose=False)
    result = results[0]
    elapsed = time.time() - start
    print(f"\n[1] 推論完了 ({elapsed*1000:.0f} ms)")
    
    if result.masks is None:
        print("  ⚠️ マスクが取得できませんでした")
        continue
    
    # マスク集約
    room_mask = np.zeros((h, w), dtype=np.uint8)
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    devices_raw = []
    for j, mask in enumerate(result.masks.data):
        cls_id = int(result.boxes.cls[j])
        cls_name = CLASS_NAMES[cls_id]
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
    
    # 部屋と壁のベクトル化
    rooms = extract_room_polygons(room_mask)
    walls = extract_wall_segments(wall_mask)
    print(f"\n[2] room ポリゴン: {len(rooms)} 個")
    print(f"[3] wall 線分: {len(walls)} 本")
    
    # 設備を部屋に紐付け(改良版)
    print(f"\n[4] 設備 → 部屋への紐付け(クラス別ルール)")
    for dev in devices_raw:
        config = DEVICE_CONFIG[dev["class"]]
        nearest = find_nearest_rooms(
            tuple(dev["center"]),
            rooms,
            max_distance=config["max_dist"],
            n=config["max_rooms"],
        )
        dev["room_ids"] = nearest
    
    # 集計
    devices_by_class = {}
    devices_by_room = {room.room_id: {} for room in rooms}
    unassigned = []
    for dev in devices_raw:
        cls = dev["class"]
        devices_by_class[cls] = devices_by_class.get(cls, 0) + 1
        if dev["room_ids"]:
            for rid in dev["room_ids"]:
                devices_by_room[rid][cls] = devices_by_room[rid].get(cls, 0) + 1
        else:
            unassigned.append(dev)
    
    print(f"  全体: {devices_by_class}")
    print(f"  未紐付け: {len(unassigned)} 個 / {len(devices_raw)} 個 "
          f"({100*len(unassigned)/max(len(devices_raw),1):.0f}%)")
    for room in rooms:
        counts = devices_by_room.get(room.room_id, {})
        if counts:
            print(f"  R{room.room_id}: {counts}")
    
    # JSON 出力
    rooms_json = []
    for room in rooms:
        room_dict = room.to_dict()
        room_dict["devices_in_room"] = [
            {"class": d["class"], "bbox": d["bbox"], "confidence": d["confidence"]}
            for d in devices_raw if room.room_id in d["room_ids"]
        ]
        rooms_json.append(room_dict)
    
    output = {
        "image": {"path": img_path.name, "width": w, "height": h},
        "model": "m1_seg_yolov8n",
        "summary": {
            "n_rooms": len(rooms),
            "n_walls": len(walls),
            "n_devices": len(devices_raw),
            "devices_by_class": devices_by_class,
            "devices_unassigned": len(unassigned),
        },
        "rooms": rooms_json,
        "walls": [w.to_dict() for w in walls],
        "devices": devices_raw,
    }
    
    json_path = OUTPUT_DIR / f"floorplan_{i}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[5] JSON 出力: {json_path}")
    
    # 可視化
    background = np.ones((h, w, 3), dtype=np.uint8) * 255
    for room in rooms:
        pts = np.array(room.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(background, [pts], (240, 230, 200))
    annotated = visualize_room_polygons(background, rooms, polygon_color=(0, 150, 0))
    annotated = visualize_walls(annotated, walls, color=(0, 0, 200), thickness=2)
    for dev in devices_raw:
        color = DEVICE_COLORS.get(dev["class"], (128, 128, 128))
        x1, y1, x2, y2 = dev["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        room_label = ",".join(f"R{r}" for r in dev["room_ids"]) if dev["room_ids"] else "?"
        label = f"{dev['class']}({room_label})"
        cv2.putText(
            annotated, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
        )
    
    vis_path = OUTPUT_DIR / f"floorplan_{i}.jpg"
    cv2.imwrite(str(vis_path), annotated)
    print(f"[6] 可視化: {vis_path}")

print(f"\n\n✅ 完了。{OUTPUT_DIR}/ で結果を確認してください。")
