import time
import torch
import cuda_operator  

def benchmark(size=int(1e7)):
    x = torch.randn(size, device='cuda', dtype=torch.float32)

    _ = cuda_operator.square(x)
    torch.cuda.synchronize()

    start = time.time()
    y_cuda = cuda_operator.square(x)
    torch.cuda.synchronize()
    t_cuda = (time.time() - start) * 1000 

    start = time.time()
    y_torch = x * x
    torch.cuda.synchronize()
    t_torch = (time.time() - start) * 1000

    max_err = (y_cuda - y_torch).abs().max().item()
    print(f"Custom CUDA operator: {t_cuda:.2f} ms")
    print(f"PyTorch x*x         : {t_torch:.2f} ms")
    print(f"Max absolute error  : {max_err:.3e}")

if __name__ == "__main__":
    benchmark()
