import os
import json
from PIL import Image

def load_controlnet_dataset(dataset_dir):
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata

def display_sample_info(dataset_dir, sample):
    image_path = os.path.join(dataset_dir, sample["image"])
    control_path = os.path.join(dataset_dir, sample["control"])
    
    image = Image.open(image_path)
    control = Image.open(control_path)
    
    print(f"Sample ID: {sample['id']}")
    print(f"Prompt: {sample['prompt']}")
    print(f"Image size: {image.size}")
    print(f"Control map size: {control.size}\n")
    return image, control

if __name__ == "__main__":
    # Change this path if needed: it should match the output_dir used in your preprocess script.
    dataset_dir = "./controlnet_datasets/controlnet_canny_dataset"  
    metadata = load_controlnet_dataset(dataset_dir)
    print(f"Loaded {len(metadata)} samples from dataset.\n")
    
    # Display information about the first 5 samples
    for sample in metadata[:5]:
        print("-" * 40)
        display_sample_info(dataset_dir, sample)
    
    print("Dataset ready for use in training or evaluation.")