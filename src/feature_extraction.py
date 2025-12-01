"""
Feature Extraction Module using OpenAI CLIP.

This module handles the extraction of semantic features from video keyframes
using the pre-trained CLIP (Contrastive Language-Image Pre-training) model.
These features are essential for the subsequent scene detection phase.
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

class FeatureExtractor:
    """
    Extracts semantic feature vectors from images using OpenAI CLIP.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        """
        Initialize the CLIP model and processor.

        Args:
            model_name: The HuggingFace model identifier for CLIP.
            device: 'cuda' or 'cpu'. If None, automatically detects availability.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing CLIP model ({model_name}) on {self.device}...")
        
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {str(e)}")
            
        self.model.eval()
        print("✓ CLIP model loaded successfully.")

    def extract_features(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract normalized feature vectors for a list of images.

        Args:
            image_paths: List of file paths to the images.
            batch_size: Number of images to process at once.

        Returns:
            np.ndarray: A matrix of shape (N, D) where N is the number of images
                        and D is the feature dimension (e.g., 512).
                        Vectors are L2 normalized.
        """
        features_list = []
        
        # Filter out invalid paths
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if len(valid_paths) != len(image_paths):
            print(f"Warning: {len(image_paths) - len(valid_paths)} image paths were invalid and skipped.")
        
        if not valid_paths:
            return np.empty((0, 512))

        # Process in batches
        for i in tqdm(range(0, len(valid_paths), batch_size), desc="Extracting features"):
            batch_paths = valid_paths[i : i + batch_size]
            
            try:
                # Load images
                images = []
                for p in batch_paths:
                    try:
                        images.append(Image.open(p).convert("RGB"))
                    except Exception as e:
                        print(f"Error reading image {p}: {e}")
                
                if not images:
                    continue

                # Preprocess and move to device
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Inference
                with torch.no_grad():
                    # Get image features
                    image_features = self.model.get_image_features(**inputs)
                
                # Normalize features (L2 norm)
                # CLIP features are not strictly normalized by default in some implementations,
                # but for cosine similarity, we want unit vectors.
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # Move to CPU and convert to numpy
                features_list.append(image_features.cpu().numpy())
                
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")

        if features_list:
            return np.concatenate(features_list, axis=0)
        else:
            return np.empty((0, 512))

    def save_features(self, features: np.ndarray, output_path: str):
        """
        Save the extracted features to a .npy file.

        Args:
            features: The numpy array of features.
            output_path: The file path to save to.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, features)
        print(f"Features saved to {output_path}")

    def load_features(self, input_path: str) -> np.ndarray:
        """
        Load features from a .npy file.
        """
        return np.load(input_path)

