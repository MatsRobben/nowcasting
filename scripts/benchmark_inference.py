import time
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from fire import Fire

from eval import load_module, setup_dataloader, setup_model

def benchmark_model(model, dataloader, device: str = "cuda", num_batches: int = 20, warmup: int = 5):
    """
    Benchmark a model wrapper (PyTorch-based or NumPy-based).
    
    Args:
        model: wrapper class (EarthformerModel, LDCastNowcastNet, SPROG)
        dataloader: torch DataLoader
        device: 'cuda' or 'cpu'
        num_batches: number of batches to benchmark
        warmup: number of warmup batches not counted in timing
    """
    times = []

    # Force device if possible
    if hasattr(model, "device"):
        orig_device = model.device
        model.device = device
        if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
            model.model.to(device)

    model.eval() if hasattr(model, "eval") else None

    for i, (x, y, _) in tqdm(enumerate(dataloader), total=num_batches, desc=f"Running on {device}"):
        if i >= num_batches:
            break

        # Warmup
        if i < warmup:
            _ = model(x, y)
            continue

        start = time.perf_counter()
        _ = model(x, y)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    mean_time = np.mean(times) if times else float("nan")

    # Restore original device
    if hasattr(model, "device"):
        model.device = orig_device
        if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
            model.model.to(orig_device)

    return mean_time

def main(config: str, num_batches: int = 20):
    cfg = OmegaConf.load(config)
    dataloader = setup_dataloader(**cfg.dataloader)

    # Load model wrapper (handles internal device)
    model = setup_model(module_path=cfg.eval.pop("model_path"), config=cfg)

    print(f"\n=== Inference Benchmark Results ({config}) ===")

    # CPU
    cpu_time = benchmark_model(model, dataloader, device="cpu", num_batches=num_batches)
    print(f"CPU: {cpu_time:.4f} s/batch")

    # GPU (if available and wrapper supports it)
    if torch.cuda.is_available():
        gpu_time = benchmark_model(model, dataloader, device="cuda", num_batches=num_batches)
        print(f"GPU: {gpu_time:.4f} s/batch")
    else:
        print("GPU not available")

if __name__ == "__main__":
    Fire(main)
