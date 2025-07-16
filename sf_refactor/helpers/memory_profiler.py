"""
Memory profiler for quantum machine learning experiments.
Lightweight profiling that works in HPC environments.
"""

import time
import os
from contextlib import contextmanager
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, memory profiling disabled")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class MemoryProfiler:
    """Lightweight memory profiler for quantum ML experiments."""
    
    def __init__(self, log_file=None, cli_test=False):
        """
        Initialize memory profiler.
        
        Args:
            log_file: Path to log file (None for no file logging)
            cli_test: If True, only log to terminal, never to file
        """
        self.log_file = log_file if not cli_test else None
        self.cli_test = cli_test
        self.start_time = time.time()
        
        if not PSUTIL_AVAILABLE:
            self.enabled = False
            return
            
        try:
            self.process = psutil.Process()
            self.enabled = True
            
            # Test if we can access memory info
            _ = self.process.memory_info()
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            self.enabled = False
            print("Warning: Cannot access process memory info, profiling disabled")
    
    def get_memory_info(self):
        """Get current memory usage in MB."""
        if not self.enabled:
            return {
                'rss_mb': 0,
                'vms_mb': 0, 
                'gpu_mb': 0,
                'timestamp': time.time() - self.start_time
            }
        
        try:
            mem = self.process.memory_info()
            return {
                'rss_mb': mem.rss / 1024 / 1024,  # Physical memory
                'vms_mb': mem.vms / 1024 / 1024,  # Virtual memory
                'gpu_mb': self._get_gpu_memory(),
                'timestamp': time.time() - self.start_time
            }
        except Exception as e:
            print(f"Warning: Error getting memory info: {e}")
            return {
                'rss_mb': 0,
                'vms_mb': 0,
                'gpu_mb': 0,
                'timestamp': time.time() - self.start_time
            }
    
    def _get_gpu_memory(self):
        """Get GPU memory usage if available."""
        if not TF_AVAILABLE:
            return 0
            
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Try to get memory info from first GPU
                gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                return gpu_info['current'] / 1024 / 1024
        except Exception:
            # Silently fail for GPU memory - it's optional
            pass
        return 0
    
    def log_memory(self, tag=""):
        """Log current memory usage."""
        if not self.enabled:
            return
            
        info = self.get_memory_info()
        
        # Format message
        if info['gpu_mb'] > 0:
            msg = f"[{info['timestamp']:.1f}s] {tag}: RAM={info['rss_mb']:.1f}MB, GPU={info['gpu_mb']:.1f}MB"
        else:
            msg = f"[{info['timestamp']:.1f}s] {tag}: RAM={info['rss_mb']:.1f}MB"
        
        # Print to terminal only in cli_test mode
        if self.cli_test:
            print(msg, flush=True)
        
        # Log to file if not cli_test and log_file is provided
        if self.log_file and not self.cli_test:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                with open(self.log_file, "a") as f:
                    f.write(f"{msg}\n")
            except Exception as e:
                # Only print error in cli_test mode to avoid cluttering HPC logs
                if self.cli_test:
                    print(f"Warning: Could not write to memory log file: {e}")
    
    @contextmanager
    def profile_block(self, name):
        """Context manager to profile a code block."""
        if not self.enabled:
            yield
            return
            
        self.log_memory(f"Before {name}")
        start_mem = self.get_memory_info()
        
        try:
            yield
        finally:
            end_mem = self.get_memory_info()
            delta = end_mem['rss_mb'] - start_mem['rss_mb']
            self.log_memory(f"After {name} (Δ{delta:+.1f}MB)")


def create_memory_profiler(cfg, run_name=None):
    """
    Factory function to create a memory profiler based on config.
    
    Args:
        cfg: Configuration object
        run_name: Run name for log file path
        
    Returns:
        MemoryProfiler instance or None if disabled
    """
    if not cfg.profiling.memory_enabled:
        return None
        
    log_file = None
    if not cfg.runtime.cli_test and run_name:
        log_file = os.path.join(cfg.data.save_dir, run_name, "memory_profile.txt")
    
    return MemoryProfiler(log_file=log_file, cli_test=cfg.runtime.cli_test)
