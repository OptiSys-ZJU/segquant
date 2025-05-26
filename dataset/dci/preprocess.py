import time
import re
from tqdm import tqdm  # To measure processing time
from backend.torch.utils import load_image
from dataset.processor import ControlNetPreprocessor


def create_dataset(
    preprocessor,
    dataset,
    output_dir,
    cn_type="canny",
    enable_no_prompt=False,
    enable_extra_caption=True,
):
    """
    Creates a ControlNet dataset by processing images from a source dataset.
    
    Args:
        preprocessor: The ControlNetPreprocessor instance
        dataset: The source dataset (e.g., COCO)
        output_dir: Directory to save the processed dataset
        cn_type: Type of control map ('canny' or 'depth')
        limit: Maximum number of samples to process (None for all)
    
    Returns:
        Path to the created dataset
    """
    import os
    import json

    # Create output directories
    dataset_dir = os.path.join(output_dir, f"dci_{cn_type}")
    images_dir = os.path.join(dataset_dir, "images")
    controls_dir = os.path.join(dataset_dir, "controls")

    os.makedirs(dataset_dir, exist_ok=True)
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
        os.path.join(dataset, "annotations"), r".*\.json"
    )

    total_samples = len(origin_data)
    print(f"Processing {total_samples} samples for ControlNet {cn_type} dataset...")

    for i, orig in enumerate(
        tqdm(origin_data, total=total_samples, desc="Processing samples"), 1
    ):
        with open(orig, "r") as file:
            d = json.load(file)

        if "image" not in d:
            raise KeyError()

        input_image = load_image(os.path.join(dataset, "photos", d["image"]))
        prompt = d.get("short_caption", None)

        if prompt is None:
            if enable_no_prompt:
                prompt = ""
            else:
                continue

        if enable_extra_caption and "extra_caption" in d:
            prompt = prompt + " " + d["extra_caption"]

        try:
            control_map = preprocessor.process(cn_type=cn_type, image=input_image)

            # Save original image and control map
            image_filename = f"image_{i:06d}.png"
            control_filename = f"control_{i:06d}.png"

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
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset created at: {dataset_dir}")
    print(f"Total processed samples: {len(metadata)}")
    return dataset_dir


if __name__ == "__main__":
    import argparse

    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Create ControlNet dataset from COCO")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./controlnet_datasets",
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
        "--limit", type=int, default=None, help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--enable_blur",
        action="store_true",
        help="Enable Gaussian blur for Canny edge detection",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="densely_captioned_images",
        help="Dataset to use (default: densely_captioned_images)",
    )
    parser.add_argument(
        "--blur_kernel_size",
        type=int,
        default=3,
        help="Kernel size used to blur the image before Canny edge detection (must be odd)",
    )
    parser.add_argument("--enable_no_prompt", action="store_true")
    parser.add_argument("--disable_extra_caption", action="store_true")
    args = parser.parse_args()

    print("Loading dataset...")
    start_time = time.time()

    # Initialize preprocessor
    preprocessor = ControlNetPreprocessor(
        enable_blur=args.enable_blur, blur_kernel_size=args.blur_kernel_size
    )

    # Create ControlNet dataset
    dataset_dir = create_dataset(
        preprocessor=preprocessor,
        dataset=args.dataset,
        output_dir=args.output_dir,
        cn_type=args.cn_type,
        enable_no_prompt=args.enable_no_prompt,
        enable_extra_caption=(not args.disable_extra_caption),
    )

    print(f"\nControlNet dataset created at: {dataset_dir}")
    print("Done!")
