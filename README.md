# Floor Plan Recognition

建築・間取り図面からの設備記号検出システム。物体検出 → 構造化データ(JSON)→ Viewer というパイプラインを、ミニチュア版で実装したプロジェクト。

## ステータス
✅ Phase 3 Exp 1 完了 (mAP@0.5 = **0.903** / Baseline から +8.8 ポイント)
🚧 Phase 3 Exp 2 進行中:長期学習(epochs=100)

## 最新スコア(Exp 1)

### Test セット全体

| 指標 | Baseline | **Exp 1 (現状最良)** |
|---|---|---|
| mAP@0.5 | 0.815 | **0.903** |
| mAP@0.5:0.95 | 0.616 | ~0.65 |

### クラス別 mAP@0.5(Exp 1 ベスト)

| クラス | mAP@0.5 |
|---|---|
| window | 0.993 🏆 |
| toilet | 0.985 🏆 |
| sink | 0.962 🏆 |
| door | 0.987 🏆 |
| staircase | 0.759 ○ |
| shower | 0.730 ○ |

## 仮説検証サイクル(Phase 3)

| 実験 | 仮説 | 結果 | ステータス |
|---|---|---|---|
| Baseline | YOLOv8n / imgsz=640 でどこまで取れるか | mAP@0.5 = 0.815 | ✅ |
| **Exp 1** | imgsz 640→1024 で小物体(shower)検出漏れが減る | mAP@0.5 = **0.903** (shower 0.42→**0.73**) | ✅ **仮説支持** |
| Exp 2 | epochs 50→100 でまだ伸びしろのあるクラスがさらに改善 | (進行中) | 🚧 |
| Exp 3 | モデル拡大(n→s)で全体精度UP、速度トレードオフ評価 | - | 計画中 |
| SAHI | タイル分割推論で小物体検出を改善 | - | 任意 |

詳細は `models/baseline_yolov8n/` および `models/exp1_highres_yolov8n/` 配下の README を参照。

## プロジェクトの背景

このプロジェクトは「図面 → ML → 構造化データ → Viewer」というフローを、個人プロジェクトとして実装することを目的としている。建築・設備図面認識システムにおける典型的なパイプラインのミニ実装。

## 技術スタック

- **言語**: Python 3.11
- **物体検出**: YOLOv8 (Ultralytics)
- **画像処理**: OpenCV, Pillow
- **学習環境**: Google Colab (T4 GPU)
- **推論・開発環境**: ローカル macOS (Apple Silicon M2 + MPS)
- **Viewer**: Streamlit(Phase 5 予定)

## データセット

### 採用データセット
[Floor Plan Annotation v1](https://universe.roboflow.com/smartapp-3jazx/floor-plan-annotation-u6whl/dataset/1) (Roboflow Universe)

### データセット選定の経緯

当初 CubiCasa5K(5000枚)の採用を検討したが、以下の理由で Roboflow Universe の代替データセットに切り替えた:
- Zenodo からのダウンロードが極端に低速(35分で17%)
- CubiCasa5K のアノテーション形式が SVG ポリゴンで、YOLO 形式への変換に工数が発生
- 本プロジェクトの目的(動くシステム + 仮説検証サイクル)の達成には、整形済データセットを使う方が時間効率が高いと判断

### データの前処理(`scripts/prepare_dataset.py`)

データ実物を観察した結果、以下の前処理を実施:

1. **クラス選定**: 元データ9クラスのうち、以下を除外
   - `room`, `wall`: 「領域」であり物体検出には不適と判断
   - `bathtub`: サンプル数極少(全体0.7%)で評価信頼性が確保できないため除外
2. **6クラスで最終化**: `door`, `shower`, `sink`, `staircase`, `toilet`, `window`
3. **train/val/test 再分割**: 元データは valid のみで test なしのため、70%/15%/15% に再分割(seed=42 で再現可能)

### データセット統計

| split | 画像数 | bbox数 |
|---|---|---|
| train | 158 | 1,807 |
| val | 33 | 408 |
| test | 36 | 409 |
| 合計 | 227 | 2,624 |

## 主要な発見

### Exp 1 から得られた洞察

**「Floor plan recognition は解像度依存度が極めて高いタスク」**

- 一般物体検出(COCO等)では imgsz=640 が標準
- しかし設備記号は小さく、相対的なピクセル数が少ない
- **間取り図のような特定ドメインでは、デフォルト設定を疑い、ハイパラを最適化することで大幅な精度UPが可能**

### 混同行列分析(Baseline)で発見した事実

shower の低精度(0.42)は「他クラスとの混同」ではなく「**検出漏れ(false negative)が91%**」が原因と判明。これにより「クラス分離の問題ではなく、特徴抽出の問題」と特定し、Exp 1 の解像度UPによる解決につながった。

## ディレクトリ構成
floor-plan-recognition/
├── data/                              # データセット(.gitignoreで除外)
│   ├── roboflow/                      # Roboflowから取得した元データ
│   └── floorplan_yolo/                # 6クラスにフィルタ後の最終データ
├── notebooks/
│   ├── 01_baseline_training.ipynb    # ベースライン学習
│   ├── 02_exp1_highres_training.ipynb # Exp 1: 高解像度
│   └── 03_exp2_long_training.ipynb   # Exp 2: 長期学習
├── scripts/                           # 単発実行スクリプト
│   ├── check_env.py                   # 環境確認
│   ├── download_roboflow.py           # データセット取得
│   ├── inspect_data.py                # データ統計確認
│   ├── visualize_data.py              # 可視化
│   └── prepare_dataset.py             # クラスフィルタ + 再分割
├── src/                               # メインソース
├── models/
│   ├── baseline_yolov8n/              # ベースライン学習結果
│   └── exp1_highres_yolov8n/          # Exp 1 学習結果
├── outputs/                           # その他出力
└── docs/

## セットアップと再現手順

```bash
# 仮想環境作成
conda create -n floorplan python=3.11 -y
conda activate floorplan

# 依存パッケージ
pip install -r requirements.txt

# Roboflow API キーを .env に設定
echo "ROBOFLOW_API_KEY=your_key_here" > .env

# データセット取得 + 前処理
python scripts/download_roboflow.py
python scripts/prepare_dataset.py

# 学習は Colab で実行(notebooks/*.ipynb)
```

## ライセンス
- データセット: CC BY 4.0 (Roboflow Universe)
- コード: MIT(予定)
