from tqdm import tqdm  # To measure processing time
from dataset.processor import ControlNetPreprocessor
from datasets import load_dataset
from PIL import Image
import os
import json
import random
import shutil


def create_dataset(
    preprocessor,
    input_dir,
    output_dir,
    samples_per_category=500,
):
    """
    Create a dataset for ControlNet from the specified dataset directory.
    Args:
        input_dir (str): Path to the input dataset directory.
        output_dir (str): Path to the output directory containing images and metadata.
    Returns:
        str: Path to the created dataset directory.
    """
    ORIGINAL_METADATA_FILE = "meta_data.json"
    CATEGORIES = [
        'animals', 'art', 'fashion', 'food', 'indoor',
        'landscape', 'logo', 'people', 'plants', 'vehicles'
    ]
    # Create output directories
    images_output_dir = os.path.join(output_dir, "images")
    controls_output_dir = os.path.join(output_dir, "controls")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(controls_output_dir, exist_ok=True)

    # Prepare metadata
    metadata = []

    # read metainfo from dataset_dir
    try:
        with open(os.path.join(input_dir, ORIGINAL_METADATA_FILE), 'r', encoding='utf-8') as f:
                original_metadata = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Original metadata file not found at {os.path.join(input_dir, ORIGINAL_METADATA_FILE)}")
        print("Please ensure the path is correct and the file exists.")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {os.path.join(input_dir, ORIGINAL_METADATA_FILE)}. Is it a valid JSON file?")
        return
    
    # Iterate through categories to sample and copy images
    id = 0
    for category in CATEGORIES:
        print(f"\nProcessing category: {category}...")

        original_category_path = os.path.join(input_dir, category)

        if not os.path.isdir(original_category_path):
            print(f"  WARNING: Original category folder not found: {original_category_path}. Skipping.")
            continue

        # List all image files in the original category folder
        try:
            all_images_in_category = [
                f for f in os.listdir(original_category_path)
            ]
        except FileNotFoundError:
             print(f"  ERROR: Could not list files in {original_category_path}. Check permissions or path.")
             continue

        if not all_images_in_category:
            print(f"  WARNING: No image files found in {original_category_path} for category {category}. Skipping.")
            continue

        print(f"  Found {len(all_images_in_category)} images in original '{category}' folder.")

        # Randomly select SAMPLES_PER_CATEGORY image filenames
        if len(all_images_in_category) < samples_per_category:
            print(f"  WARNING: Category '{category}' has only {len(all_images_in_category)} images, "
                  f"which is less than the required {samples_per_category}. Taking all available images.")
            sampled_image_filenames_with_ext = all_images_in_category
        else:
            sampled_image_filenames_with_ext = random.sample(all_images_in_category, samples_per_category)

        print(f"  Sampling {len(sampled_image_filenames_with_ext)} images for '{category}'.")
        
        for img_filename_with_ext in sampled_image_filenames_with_ext:
            img_base_filename = os.path.splitext(img_filename_with_ext)[0]
            if img_base_filename in original_metadata:
                image_filename = f"image_{id:06d}.jpg"
                src_img_path = os.path.join(original_category_path, img_filename_with_ext)
                dst_img_path = os.path.join(images_output_dir, image_filename) # Destination is now the shared folder
                
                # Copy the image file to the single 'images' folder
                try:
                    shutil.copy2(src_img_path, dst_img_path) # copy2 preserves metadata
                except Exception as e:
                    print(f"    ERROR copying {src_img_path} to {dst_img_path}: {e}")
                    continue # Skip this image if copying fails
                
                # Save original image and control map
                control_map = preprocessor.process(image=Image.open(dst_img_path))
                control_filename = f"control_{id:06d}.jpg"
                control_map.save(os.path.join(controls_output_dir, control_filename))
                
                # Get the prompt from original metadata
                prompt = original_metadata[img_base_filename].get("prompt", "")
                
                # Add to metadata
                metadata.append(
                    {
                        "id": id,
                        "prompt": prompt,
                        "image": f"images/{image_filename}",
                        "control": f"controls/{control_filename}",
                    }
                )
                id += 1
            else:
                print(f"  WARNING: Metadata key '{img_base_filename}' (from file '{img_filename_with_ext}') "
                      f"not found in original_metadata.json. Skipping this image.")

    # 4. Save new metadata
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
    parser = argparse.ArgumentParser(description="Create ControlNet dataset from datasets")
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
        default="MJHQ-30K",
        help="Dataset to use (default: MJHQ-30K)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../dataset",
        help="Dataset to use (default: COCO-Caption2017)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../dataset/controlnet_datasets",
        help="Directory to save the processed dataset",
    )
    parser.add_argument(
        "--blur_kernel_size",
        type=int,
        default=3,
        help="Kernel size used to blur the image before Canny edge detection (must be odd)",
    )
    parser.add_argument("--enable_no_prompt", action="store_true")
    return parser.parse_args()
 


if __name__ == "__main__":
    args = parse_args()
    # Initialize preprocessor
    preprocessor = ControlNetPreprocessor(
        enable_blur=args.enable_blur, blur_kernel_size=args.blur_kernel_size,cn_type=args.cn_type
    )
    # Create anno dataset
    control_dataset_dir = create_dataset(
        preprocessor=preprocessor,
        input_dir=os.path.join(args.dataset_dir, args.dataset),
        output_dir=os.path.join(args.output_dir, f"{args.dataset}-{args.cn_type}"),
        samples_per_category=500,  # Number of samples per category
    )

    print(f"\nControlNet dataset created at: {control_dataset_dir}")
    print("Done!")
