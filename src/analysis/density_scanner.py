"""
Density scanning functionality for compressing MNIST images along x and y dimensions.
"""

import numpy as np
from typing import Tuple, Union
from ..utils.common import validate_image_array


class DensityScanner:
    """
    A class for scanning and computing density profiles of images along x and y axes.
    """
    
    def __init__(self):
        """Initialize the density scanner."""
        pass
    
    def scan_x_density(self, image: np.ndarray) -> np.ndarray:
        """
        Compute density profile along x-axis (horizontal compression).
        
        Args:
            image: 2D image array
            
        Returns:
            1D array representing density along x-axis
        """
        validate_image_array(image)
        
        if len(image.shape) == 3:
            # If 3D, convert to grayscale by taking mean across channels
            image = np.mean(image, axis=2)
        
        # Sum pixel values along y-axis (vertical compression)
        x_density = np.sum(image, axis=0)
        
        return x_density
    
    def scan_y_density(self, image: np.ndarray) -> np.ndarray:
        """
        Compute density profile along y-axis (vertical compression).
        
        Args:
            image: 2D image array
            
        Returns:
            1D array representing density along y-axis
        """
        validate_image_array(image)
        
        if len(image.shape) == 3:
            # If 3D, convert to grayscale by taking mean across channels
            image = np.mean(image, axis=2)
        
        # Sum pixel values along x-axis (horizontal compression)
        y_density = np.sum(image, axis=1)
        
        return y_density
    
    def scan_both_densities(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute density profiles along both x and y axes.
        
        Args:
            image: 2D image array
            
        Returns:
            Tuple of (x_density, y_density) arrays
        """
        x_density = self.scan_x_density(image)
        y_density = self.scan_y_density(image)
        
        return x_density, y_density
    
    def normalize_density(self, density: np.ndarray, method: str = 'max') -> np.ndarray:
        """
        Normalize density profile.
        
        Args:
            density: 1D density array
            method: Normalization method ('max', 'sum', 'zscore')
            
        Returns:
            Normalized density array
        """
        if method == 'max':
            return density / np.max(density)
        elif method == 'sum':
            return density / np.sum(density)
        elif method == 'zscore':
            return (density - np.mean(density)) / np.std(density)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def batch_scan_densities(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scan densities for a batch of images.
        
        Args:
            images: Batch of 2D images (n_images, height, width)
            
        Returns:
            Tuple of (x_densities, y_densities) arrays
        """
        validate_image_array(images)
        
        if len(images.shape) != 3:
            raise ValueError("Batch images must be 3D array (n_images, height, width)")
        
        x_densities = []
        y_densities = []
        
        for img in images:
            x_density, y_density = self.scan_both_densities(img)
            x_densities.append(x_density)
            y_densities.append(y_density)
        
        return np.array(x_densities), np.array(y_densities)
    
    def get_density_stats(self, densities: np.ndarray) -> dict:
        """
        Get statistics for density profiles.
        
        Args:
            densities: Array of density profiles
            
        Returns:
            Dictionary with statistics
        """
        return {
            'mean': np.mean(densities, axis=0),
            'std': np.std(densities, axis=0),
            'min': np.min(densities, axis=0),
            'max': np.max(densities, axis=0),
            'median': np.median(densities, axis=0)
        }
