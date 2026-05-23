# Floor Plan Recognition

建築・間取り図面からの設備記号検出システム。物体検出 → 構造化データ(JSON)→ Viewer というパイプラインを、ミニチュア版で実装したプロジェクト。

## ステータス
✅ Phase 3 Exp 2 完了 (mAP@0.5 = **0.966** / Baseline から +0.151)
🚧 Phase 3 Exp 3 計画中:最新アーキテクチャ YOLO26n との比較

## 最新スコア(Exp 2)

### Test セット全体
| 指標 | Baseline | Exp 1 | **Exp 2 (現状最良)** |
|---|---|---|---|
| mAP@0.5 | 0.815 | 0.903 | **0.966** |
| mAP@0.5:0.95 | 0.616 | - | **0.734** |
| Precision | 0.823 | - | **0.959** |
| Recall | 0.733 | - | **0.936** |

### クラス別 mAP@0.5(Exp 2 ベスト)
| クラス | mAP@0.5 |
|---|---|
| toilet | 0.995 🏆 |
| door | 0.990 🏆 |
| window | 0.991 🏆 |
| sink | 0.989 🏆 |
| shower | 0.925 🏆 |
| staircase | 0.907 🏆 |

**全6クラスが mAP@0.5 ≥ 0.90 達成**。

## 仮説検証サイクル(Phase 3)

| 実験 | 設定 | 仮説 | 結果 | 状態 |
|---|---|---|---|---|
| Baseline | YOLOv8n / 640 / 50ep | デフォルト設定での性能 | mAP=0.815 | ✅ |
| Exp 1 | YOLOv8n / **1024** / 50ep | 解像度UPで小物体検出漏れが減る | mAP=**0.903** (shower 0.42→**0.73**) | ✅ **支持** |
| Exp 2 | YOLOv8n / 1024 / **100ep** | epochs増で伸びしろクラスが改善 | mAP=**0.966** (shower→**0.93**, staircase→**0.91**) | ✅ **支持** |
| Exp 3 | **YOLO26n** / 1024 / 100ep | 最新アーキテクチャでさらに改善するか | (進行中) | 🚧 |

詳細は `models/*/README.md` 参照。

## プロジェクトから得られた洞察

### 1. ドメイン特性に応じたハイパラの重要性
- COCO 等の一般物体検出では imgsz=640 が標準
- しかし**間取り図のような特定ドメインでは imgsz=1024 が劇的に有効**(shower mAP: 0.42 → 0.93)
- デフォルト設定を疑い、データの性質から逆算してハイパラを選ぶ重要性を実証

### 2. 仮説検証は混同行列から始まる
ベースラインで shower の精度が低いとき、「クラス間混同」ではなく「**検出漏れ91%**」と特定したことが、解像度UPという正しい改善方向につながった。問題の正確な切り分けが本質。

### 3. 小規模 split における val の信頼性問題
本プロジェクトでは val(33枚)と test(36枚)で大きな結果乖離(全体 0.882 vs 0.966、shower 0.47 vs 0.93)。**「小規模データセットでは val 指標を絶対視できない」**という実務上の重要な観察。

### 4. 学習時間 vs 精度のトレードオフ
| | 学習時間 | mAP@0.5 |
|---|---|---|
| Baseline | 4分 | 0.815 |
| Exp 1 | ~8分 | 0.903 |
| Exp 2 | 15分 | 0.966 |

精度の限界に近づくにつれ、追加投資に対する改善幅は逓減。トレードオフを意識した実験設計が必要。

## 技術スタック
- **言語**: Python 3.11
- **物体検出**: YOLOv8 (Ultralytics)
- **画像処理**: OpenCV, Pillow
- **学習環境**: Google Colab (T4 GPU)
- **推論・開発環境**: ローカル macOS (Apple Silicon M2 + MPS)
- **Viewer**: Streamlit(Phase 5 予定)

## データセット
[Floor Plan Annotation v1](https://universe.roboflow.com/smartapp-3jazx/floor-plan-annotation-u6whl/dataset/1) (Roboflow Universe)

### データ前処理(`scripts/prepare_dataset.py`)
1. **クラス選定**: 元データ9クラスから設備記号6クラスに絞り込み
   - 除外: `room`, `wall`(領域であり物体検出に不適), `bathtub`(サンプル数極少)
2. **最終6クラス**: `door`, `shower`, `sink`, `staircase`, `toilet`, `window`
3. **train/val/test 再分割**: 70%/15%/15%, seed=42

| split | 画像数 | bbox数 |
|---|---|---|
| train | 158 | 1,807 |
| val | 33 | 408 |
| test | 36 | 409 |

## ディレクトリ構成
floor-plan-recognition/
├── data/                          # データセット(.gitignoreで除外)
├── notebooks/
│   ├── 01_baseline_training.ipynb
│   ├── 02_exp1_highres_training.ipynb
│   └── 03_exp2_long_training.ipynb
├── scripts/                       # 単発実行スクリプト
├── models/
│   ├── baseline_yolov8n/         # 各実験の結果(README+メトリクス)
│   ├── exp1_highres_yolov8n/
│   └── exp2_long_yolov8n/
├── src/
├── outputs/
└── docs/

## セットアップと再現
```bash
conda create -n floorplan python=3.11 -y
conda activate floorplan
pip install -r requirements.txt
echo "ROBOFLOW_API_KEY=your_key_here" > .env
python scripts/download_roboflow.py
python scripts/prepare_dataset.py
# 学習は Colab で実行
```

## ライセンス
- データセット: CC BY 4.0 (Roboflow Universe)
- コード: MIT(予定)
