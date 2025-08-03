import torch
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer


def check_gpu_availability():
    """
    Check if GPU is available and show GPU information
    """
    print("🔍 GPU Availability Check")
    print("=" * 40)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        # GPU details
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)

        print(f"GPU Count: {gpu_count}")
        print(f"Current GPU: {current_device}")
        print(f"GPU Name: {gpu_name}")

        # Memory info
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3

        print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB")
        print(f"GPU Memory - Reserved: {memory_reserved:.2f}GB")
        print(f"GPU Memory - Total: {memory_total:.2f}GB")

        return True, current_device, memory_total
    else:
        print("❌ No GPU available - will use CPU")
        print("\n💡 To enable GPU:")
        print("1. Install CUDA: https://developer.nvidia.com/cuda-downloads")
        print("2. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False, None, 0


def evaluate_perplexity_gpu(text, model_id="distilgpt2", use_gpu=True):
    """
    GPU-optimized perplexity evaluation

    Args:
        text (str): Text to evaluate
        model_id (str): Model to use
        use_gpu (bool): Whether to use GPU if available
    """

    # Check GPU status
    gpu_available, device_id, gpu_memory = check_gpu_availability()

    # Determine device
    if use_gpu and gpu_available:
        device = f"cuda:{device_id}"
        print(f"🚀 Using GPU: {device}")

        # Check if model will fit in GPU memory
        model_size_estimates = {
            "distilgpt2": 0.3,
            "gpt2": 0.5,
            "gpt2-medium": 1.5,
            "Qwen/Qwen2.5-0.5B": 1.0,
            "google/gemma-2-2b": 4.0,
            "microsoft/Phi-3-mini-4k-instruct": 8.0,
            "mistralai/Mistral-7B-v0.1": 14.0
        }

        estimated_size = model_size_estimates.get(model_id, 5.0)
        if estimated_size > gpu_memory * 0.8:  # Leave 20% buffer
            print(f"⚠️  Model (~{estimated_size}GB) might not fit in GPU ({gpu_memory:.1f}GB)")
            print("🔄 Consider using a smaller model or CPU")
    else:
        device = "cpu"
        print(f"🖥️  Using CPU")

    try:
        # Method 1: Using evaluate library with GPU
        perplexity_metric = load("perplexity", module_type="metric")

        print(f"📥 Loading {model_id} on {device}...")

        results = perplexity_metric.compute(
            predictions=[text],
            model_id=model_id,
            device=device  # This will use GPU if available
        )

        print(f"✅ Perplexity calculated on {device}")
        return results['perplexities'][0]

    except Exception as e:
        print(f"❌ Error with evaluate library: {e}")

        # Fallback: Manual GPU loading
        return manual_gpu_perplexity(text, model_id, device)


def manual_gpu_perplexity(text, model_id, device):
    """
    Manual perplexity calculation with explicit GPU control
    """
    try:
        print(f"🔧 Manual loading on {device}...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with GPU optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map=device,
            low_cpu_mem_usage=True
        )
        model.eval()

        print(f"✅ Model loaded on {model.device}")

        # Tokenize and move to device
        encodings = tokenizer(text, return_tensors="pt")
        encodings = {k: v.to(device) for k, v in encodings.items()}

        # Calculate perplexity
        with torch.no_grad():
            outputs = model(encodings['input_ids'], labels=encodings['input_ids'])
            perplexity = torch.exp(outputs.loss).item()

        # Clear GPU memory
        if "cuda" in device:
            torch.cuda.empty_cache()

        return perplexity

    except Exception as e:
        print(f"❌ Manual loading failed: {e}")
        return None


def benchmark_cpu_vs_gpu(text):
    """
    Benchmark CPU vs GPU performance
    """
    import time

    print("🏁 CPU vs GPU Benchmark")
    print("=" * 40)

    model_id = "distilgpt2"  # Use small model for fair comparison

    # Test CPU
    print("\n⏱️  Testing CPU...")
    start_time = time.time()
    cpu_ppl = evaluate_perplexity_gpu(text, model_id, use_gpu=False)
    cpu_time = time.time() - start_time

    # Test GPU (if available)
    gpu_available, _, _ = check_gpu_availability()
    if gpu_available:
        print("\n⏱️  Testing GPU...")
        start_time = time.time()
        gpu_ppl = evaluate_perplexity_gpu(text, model_id, use_gpu=True)
        gpu_time = time.time() - start_time

        # Results
        print(f"\n📊 Results:")
        print(f"CPU  - Time: {cpu_time:.2f}s, Perplexity: {cpu_ppl:.2f}")
        print(f"GPU  - Time: {gpu_time:.2f}s, Perplexity: {gpu_ppl:.2f}")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x faster on GPU")
    else:
        print(f"\n📊 CPU Results:")
        print(f"Time: {cpu_time:.2f}s, Perplexity: {cpu_ppl:.2f}")

# Recommended models by GPU memory


def get_recommended_model_for_gpu():
    """
    Get model recommendations based on available GPU memory
    """
    gpu_available, device_id, gpu_memory = check_gpu_availability()

    if not gpu_available:
        return "distilgpt2"

    print(f"\n🎯 Model Recommendations for {gpu_memory:.1f}GB GPU:")

    if gpu_memory >= 16:
        recommended = "mistralai/Mistral-7B-v0.1"
        print(f"✅ High-end: {recommended}")
    elif gpu_memory >= 8:
        recommended = "microsoft/Phi-3-mini-4k-instruct"
        print(f"✅ Mid-range: {recommended}")
    elif gpu_memory >= 4:
        recommended = "google/gemma-2-2b"
        print(f"✅ Budget: {recommended}")
    else:
        recommended = "Qwen/Qwen2.5-0.5B"
        print(f"✅ Low-memory: {recommended}")

    return recommended

# Main GPU setup function


def setup_gpu_perplexity():
    """
    Complete GPU setup for perplexity evaluation
    """
    print("🚀 GPU Perplexity Evaluation Setup")
    print("=" * 50)

    # Check GPU
    gpu_available, _, _ = check_gpu_availability()

    # Get recommended model
    recommended_model = get_recommended_model_for_gpu()

    # Test function
    def test_perplexity_gpu(text):
        return evaluate_perplexity_gpu(text, recommended_model, use_gpu=gpu_available)

    return test_perplexity_gpu


# Usage examples
if __name__ == "__main__":
    sample_text = "The quick brown fox jumps over the lazy dog."

    # Check GPU status
    check_gpu_availability()

    print("\n" + "=" * 60 + "\n")

    # Test with GPU optimization
    print("🧪 Testing GPU-optimized perplexity...")
    ppl = evaluate_perplexity_gpu(sample_text, "distilgpt2", use_gpu=True)
    if ppl:
        print(f"✅ Perplexity: {ppl:.2f}")

    # Benchmark (uncomment to run)
    print("\n" + "=" * 60 + "\n")
    benchmark_cpu_vs_gpu(sample_text)

# Installation for GPU support:
"""
# For NVIDIA GPUs:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers evaluate accelerate

# Verify GPU installation:
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
"""
