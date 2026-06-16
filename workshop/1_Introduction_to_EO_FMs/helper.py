"""
Helper functions for visualization in the TerraMind workshop notebooks.

This module provides utilities for loading and visualizing geospatial raster data,
including RGB imagery, DEM data, labels, and predictions.
"""

import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def load_raster(file_path, rgb_only=False):
    """
    Load a raster file and return the data array.
    Masks nodata values (< -999) as NaN.
    
    Args:
        file_path: Path to the raster file
        rgb_only: If True, extract only RGB bands (1-3) from multi-band images
    
    Returns:
        numpy array or None if file doesn't exist
    """
    if file_path is None or not Path(file_path).exists():
        return None
    
    try:
        with rasterio.open(file_path) as src:
            data = src.read()
            nodata = src.nodata
            
            # If requesting RGB only and we have at least 3 bands
            if rgb_only and data.shape[0] >= 3:
                # Extract bands 1-3 (indices 0-2) for RGB
                rgb_data = data[0:3, :, :].astype(np.float32)
                
                # Mask nodata values (anything below -999)
                rgb_data = np.where(rgb_data < -999, np.nan, rgb_data)
                if nodata is not None:
                    rgb_data = np.where(rgb_data == nodata, np.nan, rgb_data)
                
                # Transpose to (H, W, 3) for matplotlib
                return np.transpose(rgb_data, (1, 2, 0))
            
            # If single band, return as 2D (e.g., labels, predictions)
            elif data.shape[0] == 1:
                single_band = data[0].astype(np.float32)
                # Mask nodata values
                single_band = np.where(single_band < -999, np.nan, single_band)
                if nodata is not None:
                    single_band = np.where(single_band == nodata, np.nan, single_band)
                return single_band
            
            # Default: return first band only for single-band data
            return data[0] if data.shape[0] == 1 else data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def normalize_rgb(rgb_data):
    """
    Normalize RGB data to 0-1 range for display.
    Handles NaN values (from nodata masking) properly.
    """
    if rgb_data is None:
        return None
    
    # Clip to 2nd and 98th percentiles for better contrast (ignoring NaN)
    p2, p98 = np.nanpercentile(rgb_data, (2, 98))
    
    # Normalize, keeping NaN values as NaN
    rgb_norm = np.clip((rgb_data - p2) / (p98 - p2), 0, 1)
    
    # Convert NaN to 0 for display (will appear as black)
    rgb_norm = np.nan_to_num(rgb_norm, nan=0.0)
    
    return rgb_norm


def plot_panel(ax, data, title, cmap=None, vmin=None, vmax=None, is_categorical=False, n_classes=2, is_dem=False):
    """
    Plot a single panel in the grid.
    
    Args:
        ax: Matplotlib axis
        data: Data to plot
        title: Panel title
        cmap: Colormap (optional)
        vmin, vmax: Value range (optional)
        is_categorical: Whether data is categorical
        n_classes: Number of classes for categorical data
        is_dem: Whether data is DEM (uses log scale and grayscale)
    """
    if data is None:
        ax.text(0.5, 0.5, 'Data not available', 
                ha='center', va='center', fontsize=12, color='red')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        return
    
    if len(data.shape) == 3:  # RGB image
        ax.imshow(data)
    else:  # Single band
        if is_categorical:
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=n_classes-1, interpolation='nearest')
        elif is_dem:
            # DEM visualization with log scale and grayscale
            # Mask negative and zero values for log scale
            data_masked = np.ma.masked_where(data <= 0, data)
            # Use hardcoded vmin/vmax for salt marsh elevations (typically 0.1m to 10m)
            im = ax.imshow(data_masked, cmap='gray', norm=LogNorm(vmin=0.1, vmax=10))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Elevation (m)')
        else:
            im = ax.imshow(data, cmap=cmap if cmap else 'terrain', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')

# Made with Bob
