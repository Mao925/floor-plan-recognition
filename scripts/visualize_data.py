"""データセットからランダムに数枚選んで、アノテーションを画像に描画して保存"""
import random
from pathlib import Path

import cv2
import yaml

DATA_ROOT = Path("data/roboflow/floor-plan-annotation-1")
OUT_DIR = Path("outputs/data_inspection")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# クラス情報を読み込み
with open(DATA_ROOT / "data.yaml") as f:
    config = yaml.safe_load(f)
class_names = config["names"]

# クラスごとの色を決める(BGR)
COLORS = [
    (0, 0, 255),    # bathtub: 赤
    (0, 255, 0),    # door: 緑
    (255, 0, 0),    # room: 青
    (0, 255, 255),  # shower: 黄
    (255, 0, 255),  # sink: マゼンタ
    (255, 255, 0),  # staircase: シアン
    (128, 0, 128),  # toilet: 紫
    (128, 128, 128), # wall: 灰
    (0, 128, 255),  # window: オレンジ
]

# train セットからランダムに 6 枚選択
random.seed(42)
train_images = sorted((DATA_ROOT / "train" / "images").glob("*.jpg"))
sampled = random.sample(train_images, min(6, len(train_images)))

print(f"全 {len(train_images)} 枚から 6 枚をランダムサンプリング")
print()

for img_path in sampled:
    label_path = DATA_ROOT / "train" / "labels" / (img_path.stem + ".txt")
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ 読み込み失敗: {img_path.name}")
        continue
    h, w = img.shape[:2]
    
    box_count = 0
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                # polygon 形式: 偶数個の x,y 座標(正規化済)
                if len(coords) >= 6 and len(coords) % 2 == 0:
                    points = []
                    for i in range(0, len(coords), 2):
                        x = int(coords[i] * w)
                        y = int(coords[i+1] * h)
                        points.append((x, y))
                    # ポリゴンから bbox 計算
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    color = COLORS[cls_id % len(COLORS)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, class_names[cls_id], (x1, max(y1-5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    box_count += 1
    
    out_path = OUT_DIR / f"vis_{img_path.stem[:30]}.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"  {img_path.name[:50]}... → {box_count} 個のbbox / 出力: {out_path.name}")

print()
print(f"✅ 完了。{OUT_DIR}/ に保存しました。Finder で開いて確認してください。")
print(f"   コマンド: open {OUT_DIR}")
