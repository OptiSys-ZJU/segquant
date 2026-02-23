import time
from PIL import Image
import numpy as np
import cv2
import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

# TODO: Add self.cn_type to control the control map type
# Then check the relative imports of this processor in benchmark
class ControlNetPreprocessor:
    """
    A class to preprocess images for ControlNet input (Canny edges, Depth maps).
    """

    def __init__(self, enable_blur=True, blur_kernel_size=3, cn_type="canny", device=None):
        """
        Initializes the preprocessor, loading necessary models.
        Args:
            device (str, optional): The device to run models on ('cuda', 'cpu').
                                    Defaults to 'cuda' if available, else 'cpu'.
        """
        print("Initializing ControlNetPreprocessor...")
        self.cn_type = cn_type
        # --- Device Setup ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # --- Canny Edge Setup ---
        if self.cn_type == "canny":

            self.canny_low_threshold = 100
            self.canny_high_threshold = 200
            self.canny_blur_kernel_size = blur_kernel_size  # Must be odd
            self.enable_blur = enable_blur

            assert (
                self.canny_blur_kernel_size % 2 != 0
            ), "Warning: Blur kernel size must be odd."
            print("Canny edge detector configured.")
        
        # --- Depth Estimation Setup ---
        # Using Intel's DPT model (Dense Prediction Transformer) via Hugging Face
        elif self.cn_type == "depth":
            depth_model_name = "Intel/dpt-large"  # Or try "Intel/dpt-hybrid-midas" for potentially faster/different results
            print(f"Loading depth estimation model: {depth_model_name}...")
            start_time = time.time()
            try:
                self.depth_feature_extractor = DPTFeatureExtractor.from_pretrained(
                    depth_model_name
                )
                self.depth_model = DPTForDepthEstimation.from_pretrained(depth_model_name)
                self.depth_model.to(self.device)
                self.depth_model.eval()  # Set model to evaluation mode
                print(
                    f"Depth model loaded to {self.device} in {time.time() - start_time:.2f} seconds."
                )
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
            edges = cv2.Canny(
                image_blurred, self.canny_low_threshold, self.canny_high_threshold
            )
        else:
            edges = cv2.Canny(
                image_gray, self.canny_low_threshold, self.canny_high_threshold
            )

        # ControlNet often expects edges as white lines on black background
        # Invert if needed (some models might expect black lines on white)
        # edges = 255 - edges
        return Image.fromarray(edges).convert("L")  # Ensure grayscale PIL image

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

        original_size = image.size  # W, H

        # Prepare image for the model
        inputs = self.depth_feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.depth_model(pixel_values)
            predicted_depth = outputs.predicted_depth  # This is raw output (logits)

        # Interpolate prediction to original image size
        # Note: PIL size is (W, H), interpolate expects (H, W)
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=original_size[::-1],  # Reverse to (H, W)
            mode="bicubic",
            align_corners=False,
        )

        # Normalize and format output
        output = prediction.squeeze().cpu().numpy()
        # Normalize to 0-1 range
        formatted = (output - np.min(output)) / (np.max(output) - np.min(output))
        # Scale to 0-255 and convert to uint8 grayscale image
        depth_map_np = (formatted * 255).astype(np.uint8)
        depth_map_image = Image.fromarray(depth_map_np).convert(
            "L"
        )  # Ensure grayscale PIL image

        return depth_map_image

    def process(
        self, image: Image.Image
    ) -> tuple[Image.Image | None, Image.Image | None]:
        """
        Generates both Canny edge map and depth map for the input image.
        Args:
            image (PIL.Image.Image): Input image.
        Returns:
            tuple[Image.Image | None, Image.Image | None]: (canny_map, depth_map)
        """
        if self.cn_type == "canny":
            res_map = self.get_canny_map(image)
        elif self.cn_type == "depth":
            res_map = self.get_depth_map(image)
        else:
            print("Type does not exist")
        return res_map
