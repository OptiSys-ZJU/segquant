import torch
import torch.nn.functional as F

def im2col_input(x: torch.Tensor, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    N, C, H, W = x.shape
    if C % groups != 0:
        raise ValueError(f"Input channels {C} not divisible by groups {groups}")
    C_per_group = C // groups

    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size

    X_cols = []
    for g in range(groups):
        x_group = x[:, g*C_per_group:(g+1)*C_per_group, :, :]
        x_unfold = F.unfold(
            x_group, kernel_size=(KH, KW), dilation=dilation, padding=padding, stride=stride
        )  # [batch, C_per_group*kh*kw, Hout*Wout]
        X_col_group = x_unfold.permute(0, 2, 1).reshape(-1, C_per_group*KH*KW)  # [batch*Hout*Wout, C_per_group*kh*kw]
        X_cols.append(X_col_group)

    X_col = torch.stack(X_cols, dim=0)  # [groups, batch*Hout*Wout, C_per_group*kh*kw]
    return X_col

def im2col_weight(weight: torch.Tensor, groups=1):
    # weight shape: [out, in_per_group, kh, kw]
    F_out = weight.shape[0]
    if F_out % groups != 0:
        raise ValueError(f"Output channels {F_out} not divisible by groups {groups}")
    F_per_group = F_out // groups
    W_col = weight.reshape(groups, F_per_group, -1) # [groups, out_per_group, in_per_group * kh * kw]
    return W_col

def calc_conv_output_size(H_in, W_in, kernel_size, stride=1, padding=0, dilation=1):
    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    if isinstance(dilation, int):
        dil_h = dil_w = dilation
    else:
        dil_h, dil_w = dilation

    H_out = (H_in + 2*pad_h - dil_h*(KH-1) -1)//stride_h + 1
    W_out = (W_in + 2*pad_w - dil_w*(KW-1) -1)//stride_w + 1
    return (H_out, W_out)

