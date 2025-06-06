import time
import re
import os
from tqdm import tqdm  # To measure processing time
from backend.torch.utils import load_image
from dataset.processor import ControlNetPreprocessor
import random


def create_dataset(
    preprocessor,
    dataset_dir,
    output_dir,
    enable_extra_caption=True,
    sample_size=5000,
):
    """
    Creates a ControlNet dataset by processing images from a source dataset.
    
    Args:
        preprocessor: The ControlNetPreprocessor instance
        dataset_dir: The source dataset (e.g., COCO)
        output_dir: Directory to save the processed dataset
        cn_type: Type of control map ('canny' or 'depth')
        limit: Maximum number of samples to process (None for all)
    
    Returns:
        Path to the created dataset
    """
    import os
    import json

    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    controls_dir = os.path.join(output_dir, "controls")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(controls_dir, exist_ok=True)

    # Prepare metadata
    metadata = []

    def list_matching_file_paths_regex(folder_path, pattern):
        regex = re.compile(pattern)
        return [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and regex.fullmatch(f)
        ]

    origin_data = list_matching_file_paths_regex(
        os.path.join(dataset_dir, "annotations"), r".*\.json"
    )

    total_samples = len(origin_data) if sample_size is None else min(sample_size, len(origin_data))
    origin_data = random.sample(origin_data, total_samples)
    print(f"Processing {total_samples} samples for ControlNet {preprocessor.cn_type} dataset...")

    for i, orig in enumerate(
        tqdm(origin_data, total=total_samples, desc="Processing samples"), 1
    ):
        with open(orig, "r") as file:
            d = json.load(file)

        if "image" not in d:
            raise KeyError()

        input_image = load_image(os.path.join(dataset_dir, "photos", d["image"]))
        prompt = d.get("short_caption", "")

        if enable_extra_caption and "extra_caption" in d:
            prompt = prompt + " " + d["extra_caption"]

        try:
            control_map = preprocessor.process(image=input_image)

            # Save original image and control map
            image_filename = f"image_{i:06d}.jpg"
            control_filename = f"control_{i:06d}.jpg"

            input_image.save(os.path.join(images_dir, image_filename))
            control_map.save(os.path.join(controls_dir, control_filename))

            # Add to metadata
            metadata.append(
                {
                    "id": i,
                    "prompt": prompt,
                    "image": f"images/{image_filename}",
                    "control": f"controls/{control_filename}",
                }
            )
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset created at: {output_dir}")
    print(f"Total processed samples: {len(metadata)}")
    return output_dir

def parse_args():
    import argparse
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Create ControlNet dataset from COCO")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../dataset/controlnet_datasets",
        help="Directory to save the processed dataset",
    )
    parser.add_argument(
        "--cn_type",
        type=str,
        default="canny",
        choices=["canny", "depth"],
        help="Type of control map to generate",
    )
    parser.add_argument(
        "--enable_blur",
        action="store_true",
        help="Enable Gaussian blur for Canny edge detection",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="DCI",
        help="Dataset to use (default: DCI)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../dataset/densely_captioned_images",
        help="Dataset to use (default: densely_captioned_images)",
    )
    parser.add_argument(
        "--blur_kernel_size",
        type=int,
        default=3,
        help="Kernel size used to blur the image before Canny edge detection (must be odd)",
    )
    parser.add_argument(
        "--sample_size", type=int, default=5000, help="Maximum number of samples to process"
    )
    parser.add_argument("--disable_extra_caption", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Initialize preprocessor
    preprocessor = ControlNetPreprocessor(
        enable_blur=args.enable_blur, blur_kernel_size=args.blur_kernel_size, cn_type=args.cn_type
    )

    # Create ControlNet dataset
    dataset_dir = create_dataset(
        preprocessor=preprocessor,
        dataset_dir=args.dataset_dir,
        output_dir=os.path.join(args.output_dir, f"{args.dataset}-{args.cn_type}"),
        enable_extra_caption=(not args.disable_extra_caption),
        sample_size=args.sample_size,
    )

    print(f"\nControlNet dataset created at: {dataset_dir}")
    print("Done!")