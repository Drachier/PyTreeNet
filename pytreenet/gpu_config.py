"""
This module provides a configuration system to enable/disable GPU
acceleration for QR/SVD operations throughout the library using CuPy.

When GPU acceleration enabled, all tensor_qr_decomposition() and tensor_svd() 
calls throughout the library (including BUG algorithms, canonical form 
operations, time evolution, etc.) will automatically use GPU acceleration 
for large tensors.

USAGE:
======

Enable GPU acceleration:
    from pytreenet.gpu_config import enable_gpu_acceleration
    enable_gpu_acceleration(threshold=4*1e5, verbose=True)

Disable GPU acceleration:
    from pytreenet.gpu_config import disable_gpu_acceleration
    disable_gpu_acceleration()

Check GPU status:
    from pytreenet.gpu_config import is_gpu_enabled
    print(f"GPU enabled: {is_gpu_enabled()}")

THRESHOLD BEHAVIOR:
==================

The threshold parameter controls when GPU acceleration is used:
- Tensors with size >= threshold: Use GPU (CuPy)
- Tensors with size < threshold: Use CPU (NumPy)
- GPU acceleration provides speedup primarily for large tensor operations.
  Small tensors are slower on GPU due to memory transfer overhead.
- Recommended threshold: 400,000 elements

REQUIREMENTS:
=============

GPU acceleration requires:
- NVIDIA GPU with CUDA support
- CuPy library installed (pip install cupy-cuda11x or cupy-cuda12x)

"""

class GPUConfig:
    """
    Attributes:
        use_gpu (bool): Whether GPU acceleration is enabled
        gpu_threshold (int): Minimum tensor size for GPU usage
        verbose (bool): Whether to print GPU/CPU usage messages
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_defaults()
        return cls._instance

    def _init_defaults(self):
        """Initialize default configuration values."""
        self.use_gpu = False
        self.gpu_threshold = 4*1e5  # Default threshold for GPU usage
        self.verbose = False
        self._gpu_available = None

    @classmethod
    def set_gpu_mode(cls, enabled=True, threshold=4*1e5, verbose=False):
        """
        Enable or disable GPU acceleration globally for all tensor operations.
        """
        config = cls()
        config.use_gpu = enabled
        config.gpu_threshold = threshold
        config.verbose = verbose
        
        if enabled and not config.is_gpu_available():
            print("Warning: GPU acceleration requested but CuPy/CUDA not available. Using CPU.")
            config.use_gpu = False

    @classmethod
    def is_gpu_available(cls):
        """
        Check if GPU acceleration is available.
        Returns:
            bool: True if CuPy and CUDA are available
        """
        config = cls()
        if config._gpu_available is None:
            try:
                import cupy
                config._gpu_available = cupy.cuda.is_available()
            except ImportError:
                config._gpu_available = False
        return config._gpu_available

    @property
    def gpu_available(self):
        return self.is_gpu_available()

    @classmethod
    def should_use_gpu(cls, tensor_size):
        """
        Determine if GPU should be used for a tensor of given size.
        Decision criteria:
            - GPU is enabled (use_gpu=True)
            - GPU hardware is available (CuPy/CUDA detected)
            - Tensor size >= configured threshold
        Args:
            tensor_size (int): Number of elements in the tensor to be decomposed.
        Returns:
            bool: True if GPU should be used, False if CPU should be used.
        """
        config = cls()
        return (config.use_gpu and 
                config.is_gpu_available() and 
                tensor_size >= config.gpu_threshold)

    @classmethod
    def get_info(cls):
        """Get current configuration information."""
        config = cls()
        return {
            'use_gpu': config.use_gpu,
            'gpu_threshold': config.gpu_threshold,
            'verbose': config.verbose,
            'gpu_available': config.is_gpu_available()}


def enable_gpu_acceleration(threshold=4*1e5, verbose=False):
    """
    Enable GPU acceleration globally for all QR and SVD operations in PyTreeNet.
    
    This function enables CuPy-based GPU acceleration for tensor_qr_decomposition()
    and tensor_svd() calls throughout the entire library, including:
    - Canonical form operations
    - Time evolution algorithms (TDVP, TEBD)
    - Manual tensor decompositions
    Args:
        threshold (int, optional): Minimum tensor size (number of elements) to use GPU.
            Tensors smaller than this threshold will use CPU to avoid GPU overhead.
            Defaults to 400,000.
        verbose (bool, optional): Whether to print which method (GPU/CPU) is being
            used for each operation. Useful for debugging and optimization.
            Defaults to False.
    
    Example:
        >>> enable_gpu_acceleration(threshold=10000, verbose=True)
        >>> # Now all PyTreeNet operations automatically use GPU when beneficial
        >>> # Create and run BUG algorithm - QR/SVD operations will use GPU
        >>> bug = FPBUG(initial_state=my_ttn, hamiltonian=my_ttno, ...)
        >>> bug.run()  # GPU acceleration applied automatically
        
    Note:
        Requires NVIDIA GPU with CUDA and CuPy library installed.
        If GPU is not available, operations will automatically fall back to CPU.
    """
    GPUConfig.set_gpu_mode(True, threshold, verbose)

def disable_gpu_acceleration():
    """
    Disable GPU acceleration globally for all tensor operations.
    Forces all tensor_qr_decomposition() and tensor_svd() operations 
    throughout PyTreeNet to use CPU (NumPy) instead of GPU (CuPy).
    """
    GPUConfig.set_gpu_mode(False)

def is_gpu_enabled():
    """
    Check if GPU acceleration is currently enabled and available.
    """
    return GPUConfig().use_gpu and GPUConfig.is_gpu_available()

