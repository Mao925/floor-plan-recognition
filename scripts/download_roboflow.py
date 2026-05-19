"""Roboflow Universe から間取り図データセットをダウンロード"""
import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

# .env ファイルからAPI キーを読み込む
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise RuntimeError("ROBOFLOW_API_KEY が .env に設定されていません")

print(f"API キー検出: {api_key[:3]}***{api_key[-3:]}")
print()

# ダウンロード先
output_dir = Path("data/roboflow")
output_dir.mkdir(parents=True, exist_ok=True)

# 作業ディレクトリを一時的に data/roboflow に変更
# (Roboflow SDK は カレントディレクトリにダウンロードするため)
original_cwd = os.getcwd()
os.chdir(output_dir)

try:
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("smartapp-3jazx").project("floor-plan-annotation-u6whl")
    dataset = project.version(1).download("yolov8")

    print()
    print(f"✅ ダウンロード完了!")
    print(f"   保存先: {dataset.location}")
finally:
    os.chdir(original_cwd)

# ダウンロード後の構造を確認
print()
print("=" * 60)
print("ダウンロードされた構造:")
print("=" * 60)
for path in sorted(output_dir.rglob("*"))[:30]:
    if path.is_file():
        size_kb = path.stat().st_size / 1024
        print(f"  {path.relative_to(output_dir)}: {size_kb:.1f} KB")
    else:
        print(f"  {path.relative_to(output_dir)}/ (dir)")
