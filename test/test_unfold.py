import pandas as pd
import torch
import torch.nn.functional as F

def im2col(X, kernel_size, stride=1, padding=0):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    X_col = F.unfold(X, kernel_size=kernel_size, dilation=1, padding=padding, stride=stride)
    return X_col

if __name__ == "__main__":
    B, C_in, H, W = 1, 3, 9, 9
    Kh, Kw = 3, 3
    stride = 1
    padding = 1
    
    X = torch.arange(start=1, end=B*C_in*H*W+1).float().reshape(B, C_in, H, W)

    # df = pd.DataFrame(X[0][0].to(dtype=torch.int32).numpy())
    # df.to_csv("X_0.csv")

    # df = pd.DataFrame(X[0][1].to(dtype=torch.int32).numpy())
    # df.to_csv("X_1.csv")

    # df = pd.DataFrame(X[0][2].to(dtype=torch.int32).numpy())
    # df.to_csv("X_2.csv")
    
    X_col = im2col(X, kernel_size=(Kh, Kw), stride=stride, padding=padding)[0].T
    # print(f"X_col shape: {X_col.shape}")
    # print(f"X_col: {X_col}")
    X_col = X_col[:, 0:Kh*Kw]
    print(X_col)

    df = pd.DataFrame(X_col.to(dtype=torch.int32).numpy())
    df.to_csv("X_col.csv")