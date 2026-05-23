"""ローカル(M2 + MPS)で best.pt が動作するか確認"""
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ============================================================
# 設定
# ============================================================
MODEL_PATH = "models/exp2_long_yolov8n/weights/best.pt"
TEST_IMAGES_DIR = Path("data/floorplan_yolo/test/images")
OUTPUT_DIR = Path("outputs/local_inference_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 環境確認
# ============================================================
print("=" * 60)
print("環境確認")
print("=" * 60)
print(f"PyTorch:        {torch.__version__}")
print(f"MPS available:  {torch.backends.mps.is_available()}")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用デバイス:    {device}")

# ============================================================
# モデルロード
# ============================================================
print()
print("=" * 60)
print("モデルロード")
print("=" * 60)
model_path = Path(MODEL_PATH)
if not model_path.exists():
    raise FileNotFoundError(f"モデルが見つかりません: {model_path}")

print(f"モデル: {model_path}")
print(f"サイズ: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

start = time.time()
model = YOLO(str(model_path))
print(f"ロード時間: {time.time() - start:.2f} 秒")

# ============================================================
# テスト画像を取得
# ============================================================
test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
print()
print(f"test 画像数: {len(test_images)}")
if len(test_images) == 0:
    raise RuntimeError("test 画像が見つかりません。`python scripts/prepare_dataset.py` を実行してください。")

# ============================================================
# 推論を実行(最初の3枚)
# ============================================================
print()
print("=" * 60)
print("推論実行(最初の3枚)")
print("=" * 60)

samples = test_images[:3]
for i, img_path in enumerate(samples, 1):
    print(f"\n--- 画像 {i}/3: {img_path.name[:50]}... ---")
    
    start = time.time()
    results = model.predict(
        source=str(img_path),
        imgsz=1024,             # 学習時と同じ
        conf=0.25,
        device=device,
        verbose=False,
    )
    elapsed = time.time() - start
    
    result = results[0]
    n_detections = len(result.boxes) if result.boxes is not None else 0
    print(f"検出数: {n_detections}")
    print(f"推論時間: {elapsed*1000:.0f} ms")
    
    # 検出物の詳細
    if n_detections > 0:
        class_names = result.names
        boxes = result.boxes
        for j in range(min(5, n_detections)):
            cls_id = int(boxes.cls[j])
            conf = float(boxes.conf[j])
            print(f"  - {class_names[cls_id]:10s} conf={conf:.3f}")
        if n_detections > 5:
            print(f"  ... 他 {n_detections - 5} 個")
    
    # 可視化画像を保存
    annotated = result.plot()
    out_path = OUTPUT_DIR / f"test_{i}_{img_path.stem[:30]}.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"可視化保存: {out_path}")

print()
print("=" * 60)
print(f"✅ 動作確認完了。{OUTPUT_DIR}/ に可視化画像を保存しました。")
print(f"   コマンド: open {OUTPUT_DIR}")
