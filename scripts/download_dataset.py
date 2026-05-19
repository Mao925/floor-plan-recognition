"""CubiCasa5K を HuggingFace ミラーからダウンロード"""
from huggingface_hub import snapshot_download
import os

print("Downloading CubiCasa5K from HuggingFace mirror...")
print("これは数分〜10分かかります。途中の進捗バーを確認してください。")
print()

# ダウンロード先: data/cubicasa5k_hf/
local_dir = "data/cubicasa5k_hf"
os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id="Claudio9701/cubicasa5k",
    repo_type="dataset",
    local_dir=local_dir,
)

print()
print(f"完了! ダウンロード先: {local_dir}")
print("中身を確認します:")
for item in sorted(os.listdir(local_dir))[:20]:
    path = os.path.join(local_dir, item)
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {item}: {size_mb:.1f} MB")
    else:
        print(f"  {item}/ (directory)")
