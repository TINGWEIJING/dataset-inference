import os
import shutil
from pathlib import Path

if __name__ == "__main__":
    # print(Path.cwd())
    DATASET_PARENT_DIR_PATH = Path.cwd() / "_dataset"
    RAW_DVC_TRAIN_DIR_PATH = DATASET_PARENT_DIR_PATH / "raw_dogs-vs-cats" / "train"
    DVC_TRAIN_DIR_PATH = DATASET_PARENT_DIR_PATH / "dogs-vs-cats" / "train"
    DVC_TRAIN_CAT_DIR_PATH = DATASET_PARENT_DIR_PATH / "dogs-vs-cats" / "train" / "cat"
    DVC_TRAIN_DOG_DIR_PATH = DATASET_PARENT_DIR_PATH / "dogs-vs-cats" / "train" / "dog"

    DVC_TRAIN_CAT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    DVC_TRAIN_DOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
    # check images
    cat_img_paths = []
    dog_img_paths = []

    for path in RAW_DVC_TRAIN_DIR_PATH.iterdir():
        if path.is_file():
            class_name = path.name.split(".")[0]
            if class_name == "cat":
                cat_img_paths.append(path)
            elif class_name == "dog":
                dog_img_paths.append(path)

    print(f"num cat: {len(cat_img_paths)}")
    print(f"num dog: {len(dog_img_paths)}")

    for img_path in cat_img_paths:
        shutil.copy2(
            src=img_path,
            dst=DVC_TRAIN_CAT_DIR_PATH
        )

    for img_path in dog_img_paths:
        shutil.copy2(
            src=img_path,
            dst=DVC_TRAIN_DOG_DIR_PATH
        )
