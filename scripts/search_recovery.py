import torch
from collections import deque
import numpy as np
import re

alpha_map = torch.load('search_dump_mjhq.pt', weights_only=False)

all_linear_list = []

logfile = 'best.log'
this_linear_data = {
    'opt_err': [],
    'opt_alpha': [],
}
with open(logfile, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        
        if line.startswith('Next alphas to try: '):
            value = float(line.split(":")[1].strip())
            this_linear_data['alpha'] = [value for _ in range(len(this_linear_data['opt_err']))]
            candidate_alphas = np.round(np.arange(value + 0.1, 1.0 + 1e-8, 0.1), 4).tolist()
            this_linear_data['candidate_alphas'] = [deque(candidate_alphas) for _ in range(len(this_linear_data['opt_err']))]

            print(this_linear_data)

            all_linear_list.append(this_linear_data)
            this_linear_data = {
                'opt_err': [],
                'opt_alpha': [],
            }
        elif line.startswith('Chunk '):
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            chunk_id = int(numbers[0])
            current_err = float(numbers[1])
            best_err = float(numbers[2])
            best_alpha = float(numbers[3])
            
            this_linear_data['opt_err'].append(current_err)
            this_linear_data['opt_alpha'].append(best_alpha)
        else:
            raise ValueError(f"Unexpected line format: {line}")


idx = 0
for key in alpha_map.keys():
    alpha_map[key] = all_linear_list[idx]
    idx += 1
    print(alpha_map[key])

torch.save(alpha_map, 'search_dump_mjhq_next_0.6.pt')