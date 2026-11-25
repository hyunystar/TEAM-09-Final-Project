import random
import shutil
from pathlib import Path

random.seed(42)

raw_dirs = {
    "ginkgo": Path("raw_ginkgo"),
    "not_ginkgo": Path("raw_not_ginkgo"),
}
out_root = Path("dataset")
exts = {".jpg", ".jpeg", ".png", ".webp"}

# 80% train, 10% val, 10% test 비율 예시
ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

for label, src_dir in raw_dirs.items():
    files = [p for p in src_dir.iterdir() if p.suffix.lower() in exts]
    random.shuffle(files)

    n = len(files)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    splits = {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }

    for split_name, split_files in splits.items():
        dst_dir = out_root / split_name / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in split_files:
            shutil.copy2(f, dst_dir / f.name)

print("데이터셋 분할 완료!")