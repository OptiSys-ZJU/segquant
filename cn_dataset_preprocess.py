from datasets import load_dataset
import time


import cv2
import numpy as np
from PIL import Image
import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import time # To measure processing time

class ControlNetPreprocessor:
    """
    A class to preprocess images for ControlNet input (Canny edges, Depth maps).
    """
    def __init__(self, enable_blur, device=None):
        """
        Initializes the preprocessor, loading necessary models.
        Args:
            device (str, optional): The device to run models on ('cuda', 'cpu').
                                    Defaults to 'cuda' if available, else 'cpu'.
        """
        print("Initializing ControlNetPreprocessor...")
        # --- Device Setup ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # --- Canny Edge Setup ---
        self.canny_low_threshold = 100
        self.canny_high_threshold = 200
        self.canny_blur_kernel_size = 3 # Must be odd
        self.enable_blur = enable_blur

        assert self.canny_blur_kernel_size % 2 != 0, "Warning: Blur kernel size must be odd."
        print("Canny edge detector configured.")

        # --- Depth Estimation Setup ---
        # Using Intel's DPT model (Dense Prediction Transformer) via Hugging Face
        depth_model_name = "Intel/dpt-large" # Or try "Intel/dpt-hybrid-midas" for potentially faster/different results
        print(f"Loading depth estimation model: {depth_model_name}...")
        start_time = time.time()
        try:
            self.depth_feature_extractor = DPTFeatureExtractor.from_pretrained(depth_model_name)
            self.depth_model = DPTForDepthEstimation.from_pretrained(depth_model_name)
            self.depth_model.to(self.device)
            self.depth_model.eval() # Set model to evaluation mode
            print(f"Depth model loaded to {self.device} in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"Error loading depth model: {e}")
            print("Please ensure 'transformers' and 'torch' are installed correctly.")
            # Optionally handle the error, e.g., disable depth processing
            self.depth_model = None
            self.depth_feature_extractor = None

        print("Preprocessor initialization complete.")

    def _to_numpy(self, image: Image.Image) -> np.ndarray:
        """Converts PIL Image to NumPy array (RGB)."""
        return np.array(image.convert("RGB"))

    def get_canny_map(self, image: Image.Image) -> Image.Image:
        """
        Generates a Canny edge map from the input image.
        Args:
            image (PIL.Image.Image): Input image.
        Returns:
            PIL.Image.Image: Grayscale Canny edge map.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image.")

        image_np = self._to_numpy(image)
        # 1. Convert to grayscale for Canny
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # 2. Edeg detection. Apply Gaussian Blur to reduce noise and fine details if needed
        if self.enable_blur:
            kernel_size = (self.canny_blur_kernel_size, self.canny_blur_kernel_size)
            image_blurred = cv2.GaussianBlur(image_gray, kernel_size, 0)
            edges = cv2.Canny(image_blurred, self.canny_low_threshold, self.canny_high_threshold)
        else:
            edges = cv2.Canny(image_gray, self.canny_low_threshold, self.canny_high_threshold)
            
        # ControlNet often expects edges as white lines on black background
        # Invert if needed (some models might expect black lines on white)
        # edges = 255 - edges
        return Image.fromarray(edges).convert("L") # Ensure grayscale PIL image

    def get_depth_map(self, image: Image.Image) -> Image.Image | None:
        """
        Generates a depth map from the input image using a DPT model.
        Args:
            image (PIL.Image.Image): Input image.
        Returns:
            PIL.Image.Image | None: Grayscale depth map (closer is often brighter/whiter,
                                    but depends on normalization), or None if model failed to load.
        """
        if self.depth_model is None or self.depth_feature_extractor is None:
            print("Depth model not available.")
            return None
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image.")

        original_size = image.size # W, H

        # Prepare image for the model
        inputs = self.depth_feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.depth_model(pixel_values)
            predicted_depth = outputs.predicted_depth # This is raw output (logits)

        # Interpolate prediction to original image size
        # Note: PIL size is (W, H), interpolate expects (H, W)
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=original_size[::-1], # Reverse to (H, W)
            mode="bicubic",
            align_corners=False,
        )

        # Normalize and format output
        output = prediction.squeeze().cpu().numpy()
        # Normalize to 0-1 range
        formatted = (output - np.min(output)) / (np.max(output) - np.min(output))
        # Scale to 0-255 and convert to uint8 grayscale image
        depth_map_np = (formatted * 255).astype(np.uint8)
        depth_map_image = Image.fromarray(depth_map_np).convert("L") # Ensure grayscale PIL image

        return depth_map_image

    def process(self, cn_type, image: Image.Image) -> tuple[Image.Image | None, Image.Image | None]:
        """
        Generates both Canny edge map and depth map for the input image.
        Args:
            image (PIL.Image.Image): Input image.
        Returns:
            tuple[Image.Image | None, Image.Image | None]: (canny_map, depth_map)
        """
        start_time = time.time()
        print(f"Processing image of size {image.size}...")

        if cn_type == 'canny':
            res_map = self.get_canny_map(image)
        elif cn_type == 'depth':
            res_map = self.get_depth_map(image)
        else:
            print("Type does not exist")
        
        end_time = time.time()
        print(f"Processing finished in {end_time - start_time:.2f} seconds.")
        return res_map
    

def create_controlnet_dataset(preprocessor, dataset, output_dir, cn_type='canny', limit=None):
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
    dataset_dir = os.path.join(output_dir, f"controlnet_{cn_type}_dataset")
    images_dir = os.path.join(dataset_dir, "images")
    controls_dir = os.path.join(dataset_dir, "controls")
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(controls_dir, exist_ok=True)
    
    # Prepare metadata
    metadata = []
    
    # Determine how many samples to process
    total_samples = len(dataset) if limit is None else min(limit, len(dataset))
    print(f"Processing {total_samples} samples for ControlNet {cn_type} dataset...")
    
    # Process each sample
    for i in range(total_samples):
        if i % 100 == 0:
            print(f"Processing sample {i}/{total_samples}...")
        
        # Get sample from dataset
        sample = dataset[i]
        input_image = sample['image']
        
        # Get prompt (caption)
        if 'answer' in sample and sample['answer']:
            prompt = sample['answer'][0]
        elif 'caption' in sample:
            prompt = sample['caption']
        else:
            prompt = "No caption available"
        
        # Process image with ControlNet preprocessor
        try:
            control_map = preprocessor.process(cn_type=cn_type, image=input_image)
            
            # Save original image and control map
            image_filename = f"image_{i:06d}.png"
            control_filename = f"control_{i:06d}.png"
            
            input_image.save(os.path.join(images_dir, image_filename))
            control_map.save(os.path.join(controls_dir, control_filename))
            
            # Add to metadata
            metadata.append({
                "id": i,
                "prompt": prompt,
                "image": f"images/{image_filename}",
                "control": f"controls/{control_filename}"
            })
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
    parser.add_argument("--output_dir", type=str, default="./controlnet_datasets", 
                        help="Directory to save the processed dataset")
    parser.add_argument("--cn_type", type=str, default="canny", choices=["canny", "depth"],
                        help="Type of control map to generate")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Maximum number of samples to process")
    parser.add_argument("--enable_blur", action="store_true", 
                        help="Enable Gaussian blur for Canny edge detection")
    parser.add_argument("--dataset", type=str, default="lmms-lab/COCO-Caption2017",
                        help="Dataset to use (default: COCO-Caption2017)")
    parser.add_argument("--split", type=str, default="val",
                        help="Dataset split to use (default: val)")
    args = parser.parse_args()
    
    print("Loading dataset...")
    start_time = time.time()

    # Load the dataset
    try:
        dataset = load_dataset(args.dataset, split=args.split, trust_remote_code=True)
        print(f"Dataset loaded successfully in {time.time() - start_time:.2f} seconds.")
        print(f"Number of examples: {len(dataset)}")
        print("Dataset features:", dataset.features)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have an internet connection and the dataset name is correct.")
        exit()

    # Initialize preprocessor
    preprocessor = ControlNetPreprocessor(enable_blur=args.enable_blur)
    
    # Create ControlNet dataset
    dataset_dir = create_controlnet_dataset(
        preprocessor=preprocessor,
        dataset=dataset,
        output_dir=args.output_dir,
        cn_type=args.cn_type,
        limit=args.limit
    )
    
    print(f"\nControlNet dataset created at: {dataset_dir}")
    print("Done!")

    