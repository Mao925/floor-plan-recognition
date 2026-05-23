"""
Floor Plan Detector の推論モジュール

Streamlit などの UI 層から呼び出す再利用可能な推論クラス。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# データクラス
# ============================================================
@dataclass
class Detection:
    """検出結果1件を表すデータクラス"""
    class_id: int
    class_name: str
    confidence: float
    # bbox は xyxy (左上,右下の座標、整数ピクセル)
    bbox_xyxy: tuple[int, int, int, int]
    # 正規化座標 (0-1)、JSON出力用
    bbox_normalized: tuple[float, float, float, float]
    # 中心座標とサイズ(分析用)
    center_xy: tuple[int, int]
    width: int
    height: int

    def to_dict(self) -> dict:
        """JSON 出力用の辞書"""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": self.bbox_xyxy[0],
                "y1": self.bbox_xyxy[1],
                "x2": self.bbox_xyxy[2],
                "y2": self.bbox_xyxy[3],
            },
            "bbox_normalized": {
                "x1": round(self.bbox_normalized[0], 4),
                "y1": round(self.bbox_normalized[1], 4),
                "x2": round(self.bbox_normalized[2], 4),
                "y2": round(self.bbox_normalized[3], 4),
            },
            "center": {"x": self.center_xy[0], "y": self.center_xy[1]},
            "size": {"width": self.width, "height": self.height},
        }


@dataclass
class DetectionResult:
    """1枚の画像に対する推論結果全体"""
    image_height: int
    image_width: int
    inference_time_ms: float
    detections: list[Detection]
    model_name: str

    @property
    def class_counts(self) -> dict[str, int]:
        """クラスごとの検出数"""
        counts = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    def to_json_dict(self) -> dict:
        """JSON 出力用の辞書全体"""
        return {
            "model": self.model_name,
            "image": {"width": self.image_width, "height": self.image_height},
            "inference_time_ms": round(self.inference_time_ms, 2),
            "summary": {
                "total_detections": len(self.detections),
                "by_class": self.class_counts,
            },
            "detections": [d.to_dict() for d in self.detections],
        }


# ============================================================
# 推論クラス
# ============================================================
class FloorPlanDetector:
    """間取り図の設備記号検出器
    
    使い方:
        detector = FloorPlanDetector("models/exp2_long_yolov8n/weights/best.pt")
        result = detector.predict(image_array)
        annotated_image = detector.visualize(image_array, result)
        json_data = result.to_json_dict()
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        warmup: bool = True,
    ):
        """
        Args:
            model_path: best.pt のパス
            device: 'mps' / 'cuda' / 'cpu' / None(自動判定)
            warmup: True なら初期化時にダミー推論で温める(初回推論遅延を解消)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"モデルが見つかりません: {self.model_path}")
        
        # デバイス自動判定
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        
        # モデルロード
        self.model = YOLO(str(self.model_path))
        # クラス名取得
        self.class_names = self.model.names
        self.model_name = self.model_path.parent.parent.name  # "exp2_long_yolov8n" など

        # ウォームアップ(初回推論ペナルティを先に消化する)
        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """ダミー画像で1回推論しておく(MPS グラフコンパイル等の初期コストを消化)"""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(source=dummy, device=self.device, verbose=False, imgsz=640)

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        imgsz: int = 1024,
    ) -> DetectionResult:
        """画像に対する推論を実行

        Args:
            image: BGR 画像(OpenCV 形式)または RGB 画像。numpy 配列。
            conf_threshold: 信頼度しきい値(これ未満は破棄)
            iou_threshold: NMS の IoU しきい値(これ以上重なる bbox は1つに統合)
            imgsz: 推論時の入力画像サイズ
        Returns:
            DetectionResult
        """
        h, w = image.shape[:2]

        start = time.time()
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )
        elapsed_ms = (time.time() - start) * 1000

        result = results[0]
        detections: list[Detection] = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # xyxy 座標(整数ピクセル)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # 正規化座標
                nx1, ny1, nx2, ny2 = x1 / w, y1 / h, x2 / w, y2 / h
                # 中心とサイズ
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                bw, bh = x2 - x1, y2 - y1

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=self.class_names[cls_id],
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_normalized=(nx1, ny1, nx2, ny2),
                    center_xy=(cx, cy),
                    width=bw,
                    height=bh,
                ))

        return DetectionResult(
            image_height=h,
            image_width=w,
            inference_time_ms=elapsed_ms,
            detections=detections,
            model_name=self.model_name,
        )

    def visualize(
        self,
        image: np.ndarray,
        result: DetectionResult,
        classes_to_show: Optional[list[str]] = None,
        thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """検出結果を画像に描画して返す

        Args:
            image: BGR 画像
            result: predict() の戻り値
            classes_to_show: 表示するクラス名のリスト(None なら全部)
            thickness: bbox の線の太さ
            font_scale: ラベル文字の大きさ
        Returns:
            描画済み画像(BGR)
        """
        img = image.copy()
        
        # クラスごとに色を割り当てる(BGR)
        palette = [
            (255, 0, 0),     # door: 青
            (255, 0, 255),   # shower: マゼンタ
            (0, 165, 255),   # sink: オレンジ
            (0, 255, 255),   # staircase: 黄
            (0, 128, 0),     # toilet: 緑
            (255, 192, 203), # window: ピンク
        ]
        
        for det in result.detections:
            if classes_to_show is not None and det.class_name not in classes_to_show:
                continue
            
            color = palette[det.class_id % len(palette)]
            x1, y1, x2, y2 = det.bbox_xyxy
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # ラベル背景
            label = f"{det.class_name} {det.confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(
                img,
                (x1, max(y1 - label_h - 4, 0)),
                (x1 + label_w + 4, y1),
                color, -1
            )
            cv2.putText(
                img, label, (x1 + 2, max(y1 - 4, label_h)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 1, cv2.LINE_AA
            )
        
        return img


# ============================================================
# 動作確認(モジュール単体で実行した場合)
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    model_path = "models/exp2_long_yolov8n/weights/best.pt"
    test_dir = Path("data/floorplan_yolo/test/images")
    
    print(f"モデルロード中: {model_path}")
    detector = FloorPlanDetector(model_path, warmup=True)
    print(f"使用デバイス: {detector.device}")
    print(f"モデル名: {detector.model_name}")
    print(f"クラス: {list(detector.class_names.values())}")
    
    # 適当な test 画像で動作確認
    samples = sorted(test_dir.glob("*.jpg"))[:2]
    for img_path in samples:
        img = cv2.imread(str(img_path))
        result = detector.predict(img, conf_threshold=0.25, iou_threshold=0.5)
        print(f"\n--- {img_path.name[:40]}... ---")
        print(f"  検出数: {len(result.detections)}")
        print(f"  推論時間: {result.inference_time_ms:.0f} ms")
        print(f"  クラス別: {result.class_counts}")
        
        # 可視化を保存
        annotated = detector.visualize(img, result)
        out = Path("outputs/inference_module_test") / f"out_{img_path.stem[:20]}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), annotated)
        print(f"  保存: {out}")
    
    print("\n✅ inference.py の動作確認完了")
