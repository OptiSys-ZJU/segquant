import torch

def multistep():
    steps_lst = []
    for i in range(0, 28):
        steps_lst.append(torch.load(f'perfect/int8smooth_0.1_{i}_0.pt'))

    print(steps_lst)

if __name__ == '__main__':
    multistep()