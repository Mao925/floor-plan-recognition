# Floor Plan Recognition

建築・間取り図面からの設備記号検出システム。物体検出 → 構造化データ → Viewer というパイプラインを、ミニチュア版で実装したプロジェクト。

## ステータス
✅ Phase 3 (仮説検証サイクル) 完了
🚧 Phase 4-5: Streamlit Viewer の構築

## 最終成果(Phase 3 ベスト = Exp 2)

### Test セット全体メトリクス
| 指標 | Baseline | **Best (Exp 2)** | 改善 |
|---|---|---|---|
| mAP@0.5 | 0.815 | **0.966** | **+0.151** |
| mAP@0.5:0.95 | 0.616 | **0.734** | +0.118 |
| Precision | 0.823 | 0.959 | +0.136 |
| Recall | 0.733 | 0.936 | +0.203 |

### クラス別 mAP@0.5(Best)
| クラス | mAP@0.5 |
|---|---|
| toilet | 0.995 🏆 |
| window | 0.991 🏆 |
| door | 0.990 🏆 |
| sink | 0.989 🏆 |
| shower | 0.925 🏆 |
| staircase | 0.907 🏆 |

**全6クラスが mAP@0.5 ≥ 0.90** 達成。

## 仮説検証サイクル(Phase 3 全体)

4つの実験を通じた「仮説 → 実験 → 結果 → 解釈 → 次の仮説」のサイクル:

| Exp | 設定変更 | 仮説 | 結果 | 判定 |
|---|---|---|---|---|
| Baseline | YOLOv8n / 640 / 50ep | デフォルトでどこまで取れるか | mAP=0.815, shower=0.42 | 出発点 |
| Exp 1 | imgsz **1024** | 解像度UPで小物体検出漏れが減る | mAP=**0.903**, shower=**0.73** | ✅ **仮説支持** |
| Exp 2 | epochs **100** | 学習継続で伸びしろあるクラスが改善 | mAP=**0.966**, shower=**0.93** | ✅ **仮説支持** |
| Exp 3 | model **YOLO26n** | 最新アーキテクチャで更に改善 | mAP=**0.867** (悪化) | ❌ **仮説反証** |

詳細は `models/*/README.md` 参照。

## プロジェクトから得られた洞察

### 1. ドメイン特性に応じたハイパラの重要性 (Exp 1)
- 一般物体検出(COCO等)では imgsz=640 が標準
- しかし**間取り図のような小物体中心のドメインでは imgsz=1024 が劇的に有効**(shower mAP: 0.42 → 0.73)
- デフォルト設定を疑い、データの性質から逆算する重要性

### 2. 混同行列ベースの問題切り分け (Baseline → Exp 1)
ベースラインで shower の精度が低いとき、「クラス間混同」ではなく「**検出漏れ91%**」と特定。これにより「特徴抽出の問題」と判明し、解像度UPという正しい改善方向につながった。**問題の正確な切り分けが本質**。

### 3. 学習継続の判断基準 (Exp 1 → Exp 2)
Exp 1 の学習曲線が完全に平坦化していなかったことから epochs 倍増を仮説化 → 実証。**「学習曲線を見て次の手を決める」というエビデンスベースの意思決定**を実践。

### 4. 「最新モデル = 良い」とは限らない (Exp 3 ⚠️ 最重要教訓)
YOLO26 は COCO ベンチで YOLOv8 を +3.6pt 上回るが、本タスク(227枚の小規模間取り図)では **-9.9pt 悪化**。原因として:
- 事前学習レシピとの不一致(imgsz=640 + MuSGD で事前学習)
- NMS-free 設計が小データに不利
- パラメータ数縮小(3.0M → 2.5M)

→ **「論文の数字を鵜呑みにせず、自分のデータで評価する」**という ML エンジニアの基本姿勢を実証データで確認。

### 5. 小規模 split における val の不安定さ (Exp 2 観察)
val (33枚) と test (36枚) で大きな乖離(全体 0.882 vs 0.966)。**「小規模データでは val 指標を絶対視できない」** という実務上の重要な観察。

### 6. 学習時間 vs 精度のトレードオフ
| | 学習時間 | mAP@0.5 |
|---|---|---|
| Baseline | 4分 | 0.815 |
| Exp 1 | ~8分 | 0.903 |
| Exp 2 | 15分 | 0.966 |
| Exp 3 | 17分 | 0.867 |

精度の上限に近づくほど追加投資の効率は低下。トレードオフを意識した実験設計が必要。

## 技術スタック
- **言語**: Python 3.11
- **物体検出**: YOLOv8 / YOLO26 (Ultralytics)
- **画像処理**: OpenCV, Pillow
- **学習環境**: Google Colab (Tesla T4 GPU)
- **推論・開発環境**: ローカル macOS (Apple Silicon M2 + MPS)
- **Viewer**: Streamlit (Phase 5 予定)

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
│   ├── 03_exp2_long_training.ipynb
│   └── 04_exp3_yolo26n_training.ipynb
├── scripts/                       # 単発実行スクリプト
├── models/
│   ├── baseline_yolov8n/         # 各実験の結果(README + メトリクス)
│   ├── exp1_highres_yolov8n/
│   ├── exp2_long_yolov8n/        # ⭐ 最終採用モデル
│   └── exp3_yolo26n/
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
