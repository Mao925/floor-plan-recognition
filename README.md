# Floor Plan Recognition

建築・間取り図面からの設備記号検出と部屋セグメンテーション。物体検出 + セグメンテーション → 構造化データ → **3D 家ビューア**というパイプラインのミニ実装。

## ステータス
✅ Phase 3 完了: 物体検出 mAP@0.5 = **0.966**
✅ M1 達成: セグメンテーション Mask mAP@0.5 = **0.903**, room=0.987, wall=0.683
🚧 Phase 4-6: ベクトル化 + 3D 化 + Viewer

## 達成スコアの一覧

### Phase 3: 物体検出(6 クラス)
| | Baseline | **Exp 2 (Best)** |
|---|---|---|
| mAP@0.5 | 0.815 | **0.966** |

### M1: セグメンテーション(8 クラス、room/wall 追加)
| | Box mAP@0.5 | **Mask mAP@0.5** |
|---|---|---|
| 全体 | 0.913 | **0.903** |
| room | 0.988 | **0.987** ⭐ |
| wall | 0.720 | **0.683** |

詳細は `models/*/README.md` 参照。

## プロジェクトの全体パイプライン
[1] 建築図面 (JPG/PNG)
↓
[2] ML 推論
├─ Phase 3 検出モデル: door / window / shower / sink / staircase / toilet
└─ M1 セグメンテーション: room / wall(+ 他6クラス)
↓
[3] ベクトル化 (Phase 4 / M2)
├─ room mask → ポリゴン頂点
└─ wall mask → 線分の集合
↓
[4] 3D ジオメトリ (Phase 5 / M3)
├─ 部屋を押し出して立体化
└─ ドア・窓・設備をシーンに配置
↓
[5] 3D Viewer (Phase 6 / M4)
└─ Streamlit + Three.js でブラウザに表示

## 仮説検証サイクル

| Phase | 実験 | 仮説 | 結果 |
|---|---|---|---|
| Phase 3 Baseline | YOLOv8n / 640 / 50ep | デフォルトでどこまで取れるか | mAP=0.815 |
| Phase 3 Exp 1 | imgsz 640→1024 | 解像度UPで小物体検出漏れ減 | mAP=0.903 ✅ |
| Phase 3 Exp 2 | epochs 50→100 | 学習継続でさらなる改善 | mAP=0.966 ✅ |
| Phase 3 Exp 3 | model YOLOv8n→YOLO26n | 最新モデルが優位か | mAP=0.867 ❌ (反証) |
| **M1** | task: detect→**segment** + room/wall 復活 | 3D 化に必要な領域情報を学習 | Mask mAP=**0.903** ✅ |

## 主要な発見

### 1. 解像度の効果が劇的(Phase 3 Exp 1)
imgsz 640→1024 で shower mAP: 0.42 → 0.73 → 0.93。間取り図のような小物体中心のドメインでは標準設定の見直しが必須。

### 2. 「最新モデル = 良い」とは限らない(Phase 3 Exp 3)
YOLO26 は COCO ベンチで YOLOv8 を +3.6pt 上回るが、本タスクでは -9.9pt。事前学習レシピとの不一致、NMS-free 設計が小データに不利、パラメータ縮小などが原因。論文の数字を鵜呑みにせず自分のデータで評価する重要性。

### 3. 混同行列ベースの問題切り分け
Baseline で shower 精度が低いとき、「混同」ではなく「検出漏れ91%」と特定。これにより「特徴抽出の問題」と判明し解像度UPという正しい改善方向につながった。

### 4. マルチタスク学習の恩恵(M1)
セグメンテーション学習で room の領域情報を同時に学んだ結果、shower の検出精度がむしろ向上(0.93 → 0.95)。マルチタスク学習が単タスクより有利になる興味深い事例。

### 5. クラスごとに「学習しやすさ」が異なる
- room: アノテーションが大きく綺麗 → AP 0.987
- wall: 細い線状 → AP 0.683(本質的に難しい)
- 3D 化では Phase 4 のベクトル化でノイズ除去が必要

## 技術スタック
- **言語**: Python 3.11
- **ML**: YOLOv8 / YOLOv8-seg (Ultralytics)
- **画像処理**: OpenCV, Pillow
- **学習環境**: Google Colab (Tesla T4 GPU)
- **推論・開発環境**: ローカル macOS (Apple Silicon M2 + MPS)
- **Viewer**: Streamlit + Three.js (Phase 6 予定)

## データセット
[Floor Plan Annotation v1](https://universe.roboflow.com/smartapp-3jazx/floor-plan-annotation-u6whl/dataset/1) (Roboflow Universe, CC BY 4.0)

### 前処理
- **物体検出版** (`scripts/prepare_dataset.py`): 6 クラス、bbox 形式
- **セグメンテーション版** (`scripts/prepare_dataset_seg.py`): 8 クラス、polygon 形式
- train/val/test 70%/15%/15%, seed=42(再現可能)

### データセット統計

| 版 | クラス数 | train 画像 | train インスタンス |
|---|---|---|---|
| 検出 | 6 | 158 | 1,807 |
| セグ | 8 | 163 | 2,909 |

## ディレクトリ構成
floor-plan-recognition/
├── data/                          # データセット(.gitignoreで除外)
├── notebooks/
│   ├── 01_baseline_training.ipynb     # Phase 3 Baseline
│   ├── 02_exp1_highres_training.ipynb # Phase 3 Exp 1
│   ├── 03_exp2_long_training.ipynb    # Phase 3 Exp 2 (Best detect)
│   ├── 04_exp3_yolo26n_training.ipynb # Phase 3 Exp 3
│   └── 05_m1_seg_training.ipynb       # M1 Segmentation
├── scripts/
│   ├── check_env.py
│   ├── download_roboflow.py
│   ├── prepare_dataset.py            # 検出用前処理
│   ├── prepare_dataset_seg.py        # セグ用前処理
│   └── test_inference.py
├── src/
│   └── inference.py                  # 推論モジュール
├── models/
│   ├── baseline_yolov8n/             # Phase 3 結果
│   ├── exp1_highres_yolov8n/
│   ├── exp2_long_yolov8n/             # ⭐ 検出ベスト
│   ├── exp3_yolo26n/
│   └── m1_seg_yolov8n/                # ⭐ セグメンテーション
├── outputs/
└── docs/

## セットアップと再現
```bash
conda create -n floorplan python=3.11 -y
conda activate floorplan
pip install -r requirements.txt
echo "ROBOFLOW_API_KEY=your_key_here" > .env
python scripts/download_roboflow.py
python scripts/prepare_dataset.py       # 物体検出用
python scripts/prepare_dataset_seg.py   # セグメンテーション用
# 学習は Colab で実行
```

## ライセンス
- データセット: CC BY 4.0 (Roboflow Universe)
- コード: MIT(予定)
