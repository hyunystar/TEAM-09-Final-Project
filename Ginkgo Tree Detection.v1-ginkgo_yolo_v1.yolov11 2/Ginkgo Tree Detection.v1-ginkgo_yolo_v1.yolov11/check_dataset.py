from pathlib import Path

root = Path(__file__).resolve().parent

for split in ["train", "valid", "test"]:
    images = [
        p for p in (root / split / "images").glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    labels = list((root / split / "labels").glob("*.txt"))

    missing = [
        img.name for img in images
        if not (root / split / "labels" / (img.stem + ".txt")).exists()
    ]

    empty = [
        label.name for label in labels
        if not label.read_text().strip()
    ]

    print(f"\n[{split}]")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Missing labels:", len(missing))
    print("Empty labels:", len(empty))