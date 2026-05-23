# M1: Segmentation Training (YOLOv8n-seg, 8 classes)

## 目的
3D 化に向けて、設備記号の検出だけでなく **部屋(room)と壁(wall)のセグメンテーション** を実現する。

## 設定
- **Task: segment**(物体検出 detect ではない)
- **Model: yolov8n-seg.pt**
- **Classes: 8** (door, room, shower, sink, staircase, toilet, wall, window)
- imgsz: 1024 (Phase 3 Exp 2 のベスト構成を踏襲)
- epochs: 100
- batch: 4 (セグはメモリ多消費のため)
- seed: 42

## 結果 - M1 達成条件すべてクリア

### Test セット全体
| | Box mAP@0.5 | Mask mAP@0.5 |
|---|---|---|
| 全体 | **0.913** | **0.903** |
| mAP@0.5:0.95 | 0.715 | 0.594 |

### クラス別(test)

| クラス | Box AP@0.5 | Mask AP@0.5 | 評価 |
|---|---|---|---|
| toilet | 0.993 | 0.993 | 🏆 |
| window | 0.994 | 0.969 | 🏆 |
| door | 0.990 | 0.989 | 🏆 |
| **room** | **0.988** | **0.987** | 🏆 **3D化に完璧** |
| sink | 0.955 | 0.955 | ✅ |
| shower | 0.954 | 0.954 | ✅ |
| **wall** | **0.720** | **0.683** | ✅ **使えるレベル** |
| staircase | 0.713 | 0.692 | ○ |

### M1 達成条件チェック
- ✅ 学習完走 (100 epochs)
- ✅ room Mask AP@0.5 > 0.5 → **0.987 (大幅超過)**
- ✅ wall Mask AP@0.5 > 0.4 → **0.683 (十分超過)**
- ✅ door/window などの主要クラス Box AP@0.5 > 0.9 を維持

## 学習時間
100 epochs completed in 0.433 hours (約26分、T4 GPU)

## Phase 3 (検出のみ) との比較

| 指標 | Phase 3 Exp 2 (検出) | M1 (セグ) |
|---|---|---|
| クラス数 | 6 | **8** (room, wall を追加) |
| Box mAP@0.5 | 0.966 | 0.913 |
| Mask mAP@0.5 | - | **0.903** |
| shower Box | 0.925 | **0.954** |
| sink Box | 0.989 | 0.955 |
| 学習時間 | 15分 | 26分 |

クラス数が増えてタスクが複雑化したにも関わらず、Box の精度はほぼ維持。さらに shower は M1 のほうが上回った(room の領域情報がコンテキストになっている可能性)。

## 解釈
- **room の Mask AP 0.987** は驚異的な精度。アノテーション品質と「大きな閉じた領域」というクラス特性が学習に有利。3D 化の床と壁の元データとして完璧。
- **wall Mask AP 0.683** は他クラスより低いが、これは予想通り(細い線状領域 → セグが本質的に難しい)。Phase 4 のベクトル化で **HoughLinesP + 細線化** を適用して綺麗な直線群を取り出せば十分使える。
- **shower の改善**(0.93 → 0.95)は「セグメンテーション学習が周辺コンテキスト(浴室の room 領域)も同時に学ぶことで、shower の特徴抽出が補強された」可能性。マルチタスク学習の恩恵の一例。

## 次のステップ
**Phase 4: ベクトル化(M2)**
- room mask → ポリゴン頂点(cv2.approxPolyDP)
- wall mask → 線分の集合(HoughLinesP)
- 設備(door/window/sink/toilet/etc.) → bbox の中心と回転角を推定

## ファイル
- `results.png` - 学習曲線(Box loss, Mask loss, mAP)
- `confusion_matrix.png` - 混同行列
- `Box*.png` - 物体検出の PR/F1 カーブ
- `Mask*.png` - セグメンテーションの PR/F1 カーブ
- `args.yaml`, `results.csv` - 学習設定とメトリクス
- `weights/best.pt` - 学習済みベストモデル(.gitignore で除外)
