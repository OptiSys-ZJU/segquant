import math
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

def shift_tensor_columns(tensor, k, r, stride):
    HxW, total_cols = tensor.shape
    num_groups = math.ceil(total_cols / (k * stride))
    result = torch.empty_like(tensor)

    for i in range(num_groups):
        start = i * (k * stride)
        end = (i + 1) * (k * stride)
        shift_rows = (i * r) % HxW

        if end > total_cols:
            end = total_cols

        print(f"Processing group {i}: start={start}, end={end}, total_cols={total_cols}, shift_rows={shift_rows}")
        segment = tensor[:, start:end]
        shifted = torch.roll(segment, shifts=shift_rows, dims=0)
        result[:, start:end] = shifted

    return result

def conv_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

if __name__ == "__main__":
    B, C_in, H, W = 1, 3, 7, 7
    Kh, Kw = 5, 5
    stride = 2
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

    r = conv_output_size(H, Kw, stride=stride, padding=padding)
    print(r)

    X_col_shifted = shift_tensor_columns(X_col, Kw, r, stride)
    df = pd.DataFrame(X_col_shifted.to(dtype=torch.int32).numpy())
    df.to_csv("X_col_shifted.csv")

