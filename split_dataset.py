import os
import random
import shutil
import argparse
import sys

# Replace brittle hard-coded windows strings with normalized defaults relative to this script
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE = os.path.normpath(os.path.join(PROJECT_DIR, "dataset", "archive (2)"))
DEFAULT_TRAIN = os.path.normpath(os.path.join(PROJECT_DIR, "dataset", "train"))
DEFAULT_TEST = os.path.normpath(os.path.join(PROJECT_DIR, "dataset", "test"))

def find_candidate_source(default_path, project_dir, min_images=5, max_search_depth=2):
    """Try a few heuristics to locate a dataset folder when default_path is missing."""
    # 1) If default exists already, return it
    if os.path.isdir(default_path):
        return os.path.abspath(default_path)

    # 2) Try simple name variations (space <-> underscore)
    alt = default_path.replace("_", " ")
    if os.path.isdir(alt):
        return os.path.abspath(alt)
    alt2 = default_path.replace(" ", "_")
    if os.path.isdir(alt2):
        return os.path.abspath(alt2)

    # 3) Look in project parent folders for a directory containing many images
    def has_enough_images(d):
        try:
            files = os.listdir(d)
        except Exception:
            return False
        imgs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return len(imgs) >= min_images

    # check siblings and children up to a small depth
    searched = set()
    to_check = [project_dir]
    parent = os.path.dirname(project_dir)
    if parent and parent not in to_check:
        to_check.append(parent)

    for base in to_check:
        for root, dirs, files in os.walk(base):
            if root in searched:
                continue
            searched.add(root)
            if has_enough_images(root):
                return os.path.abspath(root)
            # limit depth by comparing path lengths
            if root.count(os.sep) - base.count(os.sep) >= max_search_depth:
                # skip deeper
                dirs[:] = []

    return None

def split_dataset(source_dir, train_dir, test_dir, split_ratio=0.8):
    source_dir = os.path.abspath(os.path.normpath(source_dir))
    train_dir = os.path.abspath(os.path.normpath(train_dir))
    test_dir = os.path.abspath(os.path.normpath(test_dir))

    if not os.path.isdir(source_dir):
        # Attempt to auto-detect a likely dataset location
        candidate = find_candidate_source(source_dir, PROJECT_DIR)
        if candidate:
            print(f"Default source not found. Using discovered dataset directory: {candidate}")
            source_dir = candidate
        else:
            print(f"Source directory does not exist: {source_dir}")
            print("Hints:")
            print(" - Ensure the folder path is correct. If you edited the script, use raw strings for Windows paths, e.g. r'C:\\path\\to\\dataset'")
            print(" - Or run the script with --source \"C:\\full\\path\\to\\dataset\"")
            print("The script searched common locations under the project folder for directories containing image files.")
            return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files = os.listdir(source_dir)
    images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print(f"No images found in source directory: {source_dir}")
        return

    random.shuffle(images)

    train_size = int(len(images) * split_ratio)
    train_images = images[:train_size]
    test_images = images[train_size:]

    for img in train_images:
        shutil.copy2(os.path.join(source_dir, img), train_dir)

    for img in test_images:
        shutil.copy2(os.path.join(source_dir, img), test_dir)

    print("Dataset split completed!")
    print(f"Training images: {len(train_images)}")
    print(f"Testing images: {len(test_images)}")

def main(argv):
    parser = argparse.ArgumentParser(description="Split images into train/test.")
    parser.add_argument("--source", "-s", default=DEFAULT_SOURCE, help="Source directory with images")
    parser.add_argument("--train", "-t", default=DEFAULT_TRAIN, help="Train output directory")
    parser.add_argument("--test", "-e", default=DEFAULT_TEST, help="Test output directory")
    parser.add_argument("--split", type=float, default=0.8, help="Train split ratio (0.0-1.0)")
    args = parser.parse_args(argv)

    split_dataset(args.source, args.train, args.test, split_ratio=args.split)

if __name__ == "__main__":
    main(sys.argv[1:])
