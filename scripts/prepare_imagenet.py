import argparse
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

ARCHIVES = {
    "train": "ILSVRC2012_img_train.tar",
    "val": "ILSVRC2012_img_val.tar",
    "devkit": "ILSVRC2012_devkit_t12.tar.gz",
}

def extract_files(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for member in archive:
        target = (root / member.name).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Archive entry escapes the destination: {member.name}")
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        else:
            raise ValueError(f"Unsupported archive entry: {member.name}")

def extract_training(archive_path, destination):
    classes = 0
    with tarfile.open(archive_path, mode="r|*") as archive:
        for member in archive:
            if member.isdir():
                continue
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or len(name.parts) != 1
                or not re.fullmatch(r"n\d{8}\.tar", name.name)
            ):
                raise ValueError(f"Unexpected training archive entry: {member.name}")
            with archive.extractfile(member) as class_file:
                with tarfile.open(fileobj=class_file, mode="r|*") as class_archive:
                    extract_files(class_archive, destination / name.stem)
            if not any((destination / name.stem).glob("*.JPEG")):
                raise ValueError(f"No training images found in class {name.stem}.")
            classes += 1
            if classes % 50 == 0:
                print(f"  Extracted {classes} training classes", flush=True)
    if classes == 0:
        raise ValueError("The training archive contains no class archives.")
    return classes

def prepare_imagenet(archives_dir, datasets_dir):
    archives = {split: archives_dir / name for split, name in ARCHIVES.items()}
    for path in archives.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing archive: {path}")
        with tarfile.open(path, mode="r:*"):
            pass

    destination = datasets_dir / "imagenet"
    destination.mkdir(parents=True, exist_ok=False)

    print("Extracting the development kit...", flush=True)
    with tarfile.open(archives["devkit"], mode="r|*") as archive:
        extract_files(archive, destination)
    devkit_data = destination / "ILSVRC2012_devkit_t12" / "data"
    for name in ("meta.mat", "ILSVRC2012_validation_ground_truth.txt"):
        if not (devkit_data / name).is_file():
            raise FileNotFoundError(f"Missing devkit file: {devkit_data / name}")

    print("Extracting training images by class...", flush=True)
    classes = extract_training(archives["train"], destination / "train")

    print("Extracting validation images into val/...", flush=True)
    with tarfile.open(archives["val"], mode="r|*") as archive:
        extract_files(archive, destination / "val")
    if not any((destination / "val").glob("*.JPEG")):
        raise ValueError("No validation images found directly inside val/.")

    print(f"Ready: {destination.resolve()} ({classes} training classes)")
    print(f"Set data.datasets_path to: {datasets_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archives-dir", type=Path, required=True,
        help="Directory containing the three downloaded ImageNet 2012 archives.",
    )
    parser.add_argument(
        "--datasets-dir", type=Path, required=True,
        help="Parent directory where a new imagenet/ folder will be created.",
    )
    args = parser.parse_args()
    try:
        prepare_imagenet(args.archives_dir.expanduser(), args.datasets_dir.expanduser())
    except (OSError, tarfile.TarError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

if __name__ == "__main__":
    main()
