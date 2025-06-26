import numpy as np
from scipy.ndimage import gaussian_filter



def apply_mri_variation(volume: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Applies a simulated low-frequency bias field to MRI volume.

    # References:
    # Simkó et al. (2022) – “MRI bias field correction with an implicitly trained CNN”
    # Tustison et al. (2010) – “N4ITK: Improved N3 bias correction”
    # Vagni et al. (2024) – “Impact of bias field correction on 0.35 T pelvic MR images”
    
    Args:
        volume (np.ndarray): 3D MRI volume.
        strength (float): Bias strength between 0 (no effect) and 1 (strong bias).
        
    Returns:
        np.ndarray: MRI volume with simulated bias field.
    """
    assert 0 <= strength <= 1, "Strength must be in [0, 1]"
    
    # Generate random low-frequency field
    field = np.random.rand(*volume.shape)
    sigma = tuple(s * (0.2 + 0.3 * strength) for s in volume.shape)  # smoothness depends on strength
    bias_field = gaussian_filter(field, sigma=sigma)
    
    # Normalize to multiplicative range ~[1 - α, 1 + α]
    alpha = 0.1 + 0.4 * strength
    bias_field = 1 + (bias_field - np.mean(bias_field)) / np.std(bias_field) * alpha
    biased_volume = volume * bias_field
    
    return biased_volume


def apply_ct_variation(volume: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Applies HU offset and blur to simulate calibration/reconstruction variability in CT.
    
    # References:
    # Stahl et al. / “Shifted Windows for Deep Learning Augmentation of CT Images” (2023, arXiv)

    Args:
        volume (np.ndarray): 3D CT volume in Hounsfield Units.
        strength (float): Variation strength between 0 (no effect) and 1 (strong effect).
        
    Returns:
        np.ndarray: Modified CT volume.
    """
    assert 0 <= strength <= 1, "Strength must be in [0, 1]"
    
    # Simulate HU calibration shift
    hu_shift = np.random.uniform(-30, 30) * strength
    shifted_volume = volume + hu_shift
    
    # Simulate smoothing (reconstruction kernel variation)
    sigma = 0.5 + 2.0 * strength  # larger sigma = more blur
    smoothed_volume = gaussian_filter(shifted_volume, sigma=sigma)
    
    return smoothed_volume
