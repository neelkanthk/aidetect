import torch
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer


class GpuUtility:
    def check_gpu_availability():
        """
        Check if GPU is available and return it's information
        """

        # Check CUDA availability
        is_cuda_available = torch.cuda.is_available()

        if is_cuda_available:
            # GPU details
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)

            # Memory info
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
            memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3

            return {
                "cuda_available": True,
                "device_id": current_device,
                "gpu_name": gpu_name,
                "gpu_count": gpu_count,
                "memory_allocated": memory_allocated,
                "memory_reserved": memory_reserved,
                "memory_total": memory_total
            }
        else:
            return {
                "cuda_available": False,
                "device_id": None,
                "gpu_name": None,
                "gpu_count": 0,
                "memory_allocated": 0,
                "memory_reserved": 0,
                "memory_total": 0
            }
