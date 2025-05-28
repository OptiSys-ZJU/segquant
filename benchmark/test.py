import torch
import time

def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    # Create two random matrices on GPU
    a = torch.randn(5096, 5096, device="cuda:6")
    b = torch.randn(5096, 5096, device="cuda:6")
    c = torch.randn(5096, 5096, device="cuda:7")
    d = torch.randn(5096, 5096, device="cuda:7")

    print("Starting GPU memory monitoring loop. Press Ctrl+C to stop.")

    try:
        while True:
            # Perform matrix multiplication
            e = torch.mm(a, b)
            f = torch.mm(c, d)
            
            # Get memory stats
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # in GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # in GB
            
            print(f"Allocated: {allocated:.4f} GB | Reserved: {reserved:.4f} GB")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()