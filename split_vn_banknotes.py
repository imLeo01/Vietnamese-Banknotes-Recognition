import os, shutil, re
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]

def normalize_label(name: str) -> str:
    """Chuẩn hóa tên mệnh giá: '10k' -> '10000' """
    s = name.strip().lower().replace("_", "").replace(" ", "")
    digits = re.findall(r'\d+', s)
    if not digits:
        return name
    number = int("".join(digits))
    if "k" in s:
        number *= 1000
    return str(number)

def copy_with_rename(src: Path, dst: Path):
    """Copy file, nếu trùng tên thì thêm hậu tố _1, _2..."""
    dst_file = dst / src.name
    if dst_file.exists():
        base, ext = os.path.splitext(src.name)
        k = 1
        while (dst / f"{base}_{k}{ext}").exists():
            k += 1
        dst_file = dst / f"{base}_{k}{ext}"
    shutil.copy2(src, dst_file)

def merge_into_existing(src_roots, existing_data_root):
    existing_data_root = Path(existing_data_root)
    splits = ["train", "val", "test"]

    for src_root in src_roots:
        src_root = Path(src_root)
        class_dirs = [d for d in src_root.iterdir() if d.is_dir()]
        for d in class_dirs:
            lb = normalize_label(d.name)
            files = list_images(d)
            # phân chia ngẫu nhiên files theo tỷ lệ giống dataset cũ
            n = len(files)
            n_train = int(n * 0.8)
            n_val   = int(n * 0.1)
            train_files = files[:n_train]
            val_files   = files[n_train:n_train+n_val]
            test_files  = files[n_train+n_val:]

            for split, fs in zip(splits, [train_files, val_files, test_files]):
                dst_dir = existing_data_root / split / lb
                dst_dir.mkdir(parents=True, exist_ok=True)
                for f in fs:
                    copy_with_rename(f, dst_dir)
            print(f"[{lb}] +{len(files)} ảnh (train={len(train_files)}, val={len(val_files)}, test={len(test_files)})")

    print("\n✔ Done. Dataset đã được merge vào:", existing_data_root)

if __name__ == "__main__":
    sources = [
        r"C:\Users\Lazycat\Downloads\archive (1)\dataset",   # dataset mới 1
        r"C:\Users\Lazycat\Downloads\currency_vietnam_recognition.v1i.folder\train"    # dataset mới 2
    ]
    existing = r"C:\Users\Lazycat\Documents\AI\Number recognition\banknotes\data" # dataset đã chia train/val/test
    merge_into_existing(sources, existing)
