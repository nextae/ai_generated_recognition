import os
import shutil
from pathlib import Path

REAL = ["lsun", "imagenet", "coco", "afhq", "celebahq", "ffhq", "lama??", "landscape"]

NOTES = ["mat - przerobiona lama?",
         "lama - (powtórki, jakieś cosie, może lepiej nie)",
         "metfaces - real ale sztuka a nie zdjęcia",
         "pro-gan naraz real i fake",
         "stylegan3 - jakoś dziwnie pomieszane",
         "cycle-gan - wymieszane real i fake"]

# Set the paths for Kaggle datasets
input_folder = "../datasets/datasets"
temporary_folder = "../datasets/temp"
real_images_folder = "../datasets/artifact/real_images"
fake_images_folder = "../datasets/artifact/fake_images"

# List of real image file names
real_image_files = ["afhq", "celebahq", "coco", "ffhq", "imagenet", "landscape", "lsun", "metfaces"]
to_ignore = ["lama", "pro-gan", "stylegan3", "cycle_gan"]


def copy_images(source_dir, real_target_dir, fake_target_dir):
    # Create the target directory if it doesn't exist
    Path(real_target_dir).mkdir(parents=True, exist_ok=True)
    Path(fake_target_dir).mkdir(parents=True, exist_ok=True)

    counter = 0
    # Iterate through all files and directories in the source directory
    for root, dirs, files in os.walk(source_dir):
        if any(ignore in root for ignore in to_ignore):
            continue
        for file in files:
            if file.endswith(".zip"):
                continue
            if any(real_image_file in root for real_image_file in real_image_files):
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(real_target_dir, file)
            else:
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(fake_target_dir, file)

            # Copy the image file to the target directory
            file_format = target_file_path.split(".")[1]
            root_split = root.split("\\")[1]

            shutil.copy2(source_file_path, target_file_path.split(".")[0]+"-"+root_split+"-"+str(counter)+"."+file_format)
            # print(f"Copied {source_file_path} to {target_file_path}")
        print(root)
        counter += 1


copy_images(input_folder, real_images_folder, fake_images_folder)

