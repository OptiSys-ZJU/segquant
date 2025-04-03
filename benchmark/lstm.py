import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import argparse
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

def frobenius(A, B):
    return torch.norm(A - B, p='fro')

# 创建数据集
class TimeSeriesMatrixDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_layers=2, output_dim=None):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim else input_dim

        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.output_dim)  # 线性层输出预测

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM 输出
        output = self.fc(lstm_out[:, -1, :])  # 取最后时间步输出
        return output
    

# 重新构造 LSTM 需要的 (batch, seq_len, feature_dim) 格式
def create_sequences(a_list, b_list, window_size):
    X, Y = [], []
    for i in range(len(a_list) - window_size):
        X.append(a_list[i:i+window_size])  # 过去 window_size 个 a
        Y.append(b_list[i+window_size])  # 目标 b
    return np.array(X), np.array(Y)

def multi_single(max_steps=200, quant_type='int8_smooth', ctrl_type='canny'):
    indices = list(range(max_steps))  # x 轴对应的 i 值
    timestamps = []

    singles = []
    multis = []

    for i in indices:
        for scale in [0.8]:
            fp16_all_multi = torch.load(f'multistep/fp16_{ctrl_type}_{scale}_{i}.pt')
            int8_all_multi = torch.load(f'multistep/{quant_type}_{ctrl_type}_{scale}_{i}.pt')

            timestamp = fp16_all_multi['timestep'][0].item()
            timestamps.append(timestamp)

            # error_multi = error(fp16_all_multi['noise_pred'], int8_all_multi['noise_pred'], type)
            mat_error_multi = fp16_all_multi['noise_pred'] - int8_all_multi['noise_pred']
            multi_vec = mat_error_multi.cpu().detach().flatten().numpy()
            multis.append(multi_vec)

            fp16_all_single = torch.load(f'multistep/fp16_{ctrl_type}_{scale}_{i}.pt')
            int8_all_single = torch.load(f'singlestep/{quant_type}_{ctrl_type}_{scale}_{i}.pt')
            
            
            # error_single = error(fp16_all_single['noise_pred'], int8_all_single['noise_pred'], type)
            mat_error_single = fp16_all_single['noise_pred'] - int8_all_single['noise_pred']
            single_vec = mat_error_single.cpu().detach().flatten().numpy()
            singles.append(single_vec)
    

    # 设定时间窗口大小
    window_size = 10
    alist = np.array(singles)  # shape = (T, M, N)
    blist = np.array(multis)  # shape = (T, M, N)

    X, Y = create_sequences(alist, blist, window_size)  # X: (190, 10, 524288), Y: (190, 524288)

    # 转换为 PyTorch Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    # 重新 reshape 使其符合 LSTM 输入格式
    batch_size = 32
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    input_dim = 524288  # 每个时间步的输入特征数
    model = LSTMModel(input_dim=input_dim, output_dim=input_dim).to('cuda:0')

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to('cuda:0')
            Y_batch = Y_batch.to('cuda:0')
            optimizer.zero_grad()
            Y_pred = model(X_batch)  # 预测
            loss = criterion(Y_pred, Y_batch)  # 计算误差
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.6f}")

    print("训练完成！")
    torch.save(model.state_dict(), "lstm_model.pth")  # 仅保存参数

    # 预测
    with torch.no_grad():
        Y_pred = model(X_tensor.to('cuda:0')).cpu().numpy()

    # 选取某个时间步
    t_index = -1  # 最后一个时间步
    true_b = Y_tensor[t_index].numpy()
    pred_b = Y_pred[t_index]

    print(true_b-pred_b)

if __name__ == '__main__':
    multi_single()