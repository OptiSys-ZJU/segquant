import time
import os
import random
from tqdm import tqdm  # To measure processing time
from dataset.processor import ControlNetPreprocessor
from datasets import load_dataset


def create_dataset(
    preprocessor,
    dataset,
    output_dir,
    enable_no_prompt=False,
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
    dataset_dir = output_dir
    images_dir = os.path.join(dataset_dir, "images")
    controls_dir = os.path.join(dataset_dir, "controls")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(controls_dir, exist_ok=True)

    # Prepare metadata
    metadata = []

    total_samples = len(dataset)
    print(f"Processing {total_samples} samples for ControlNet {preprocessor.cn_type} dataset...")
    # Process each sample
    for i, sample in enumerate(
        tqdm(dataset, total=total_samples, desc="Processing samples")
    ):
        input_image = sample["image"]
        if "answer" in sample and sample["answer"]:
            prompt = ""
            for ans in sample["answer"]: 
                prompt += ans
        elif "caption" in sample:
            prompt = sample["caption"]
        else:
            prompt = None

        if prompt is None:
            if enable_no_prompt:
                prompt = ""
            else:
                continue

        # Process image with ControlNet preprocessor
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
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset created at: {dataset_dir}")
    print(f"Total processed samples: {len(metadata)}")
    return dataset_dir

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
        "--sample_size", type=int, default=5000, help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--enable_blur",
        action="store_true",
        help="Enable Gaussian blur for Canny edge detection",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="COCO-Caption2017",
        help="Dataset to use (default: COCO-Caption2017)",
    )
    parser.add_argument(
        "--split", type=str, default="val", help="Dataset split to use (default: val)"
    )
    parser.add_argument(
        "--blur_kernel_size",
        type=int,
        default=3,
        help="Kernel size used to blur the image before Canny edge detection (must be odd)",
    )
    parser.add_argument("--enable_no_prompt", action="store_true")
    parser.add_argument("--random_sample", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("Loading dataset...")

    # Load the dataset
    try:
        # determine the total number of samples
        dataset = load_dataset(os.path.join("../dataset", args.dataset), split=args.split, trust_remote_code=True)
        # random sample
        if args.sample_size is not None and args.random_sample:
            total_samples = min(args.sample_size, len(dataset))
            indices = random.sample(range(len(dataset)), total_samples)
            dataset = dataset.select(indices)  # HuggingFace way
        elif args.sample_size is not None:
            dataset = dataset.select(range(args.sample_size))
            
        dataset.name = args.dataset
        print(f"Number of examples: {len(dataset)}")
        print("Dataset features:", dataset.features)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(
            "Please ensure you have an internet connection and the dataset name is correct."
        )
        exit()

    # Initialize preprocessor
    preprocessor = ControlNetPreprocessor(
        enable_blur=args.enable_blur, blur_kernel_size=args.blur_kernel_size, cn_type=args.cn_type
    )

    # Create ControlNet dataset
    dataset_dir = create_dataset(
        preprocessor=preprocessor,
        dataset=dataset,
        output_dir=os.path.join(args.output_dir, f"{args.dataset}-{args.cn_type}"),
        enable_no_prompt=args.enable_no_prompt,
    )

    print(f"\nControlNet dataset created at: {dataset_dir}")
    print("Done!")
