import torch
import numpy as np
from tqdm import trange
from pathlib import Path

device = torch.device("cuda:1")

all_vectors = np.load("/home/chenhaojun/turbozero_torch/24D_196560_1.npy")
all_vectors = torch.from_numpy(all_vectors).to(device)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

initial_actions = torch.arange(all_vectors.shape[0])
seed_list = [42]
for initial_action in initial_actions:
    for seed in seed_list:
        set_seed(seed)
        board = torch.zeros(all_vectors.shape[0], device=device, dtype=torch.bool)
        legal_actions = torch.ones(all_vectors.shape[0], device=device, dtype=torch.bool)
        N, dim = all_vectors.shape                                                                                                                                                                                                                                                      
        valid_cos_arr = torch.tensor([0, 0.25, -0.25], device=device)                                                                                                

        save_dir = Path(Path(__file__).parent, "greedy_search_results_duijing", f"seed_{seed}", f"initial_action_{initial_action}")
        save_dir.mkdir(parents=True, exist_ok=True)

        def find_valid_rows(cos_matrix, valid_cos_arr, tolerance=1e-5):
            # 使用广播和向量化操作
            diff = torch.abs(cos_matrix.unsqueeze(-1) - valid_cos_arr)
            is_valid = torch.any(diff < tolerance, dim=-1)
            return torch.all(is_valid, dim=-1)

        def count_valid_rows(cos_matrix, valid_cos_arr, tolerance=1e-5):
            diff = torch.abs(cos_matrix.unsqueeze(-1) - valid_cos_arr)
            is_valid = torch.any(diff < tolerance, dim=-1)
            valid_count =  torch.sum(is_valid, dim=-1)
            return valid_count


        while legal_actions.any():
            if not board.any():
                # action = torch.randint(0, N, (1,), device=device)
                action = initial_action
                board[action] = True
                legal_actions[action] = False
            else:
                selected_indices = torch.where(board)[0]
                unselected_indices = torch.where(legal_actions)[0]
                selected_vectors = all_vectors[selected_indices]
                unselected_vectors = all_vectors[unselected_indices]
                
                cos_matrix = unselected_vectors @ selected_vectors.T
                valid_rows = find_valid_rows(cos_matrix, valid_cos_arr)
                if not valid_rows.any():
                    break 
                legal_actions[unselected_indices[~valid_rows]] = False
                valid_cos_indices = unselected_indices[valid_rows]
                # print(f"unselected_indices: {unselected_indices.shape}, valid_cos_indices: {valid_cos_indices.shape}, valid_cos_indices.max:{valid_cos_indices.max()}")
                valid_cos_unselected_vectors = unselected_vectors[valid_rows]
                print(f"valid_cos_unselected_vectors: {valid_cos_unselected_vectors.shape}")

                count = torch.zeros(valid_cos_unselected_vectors.shape[0], device=device)
                batch_size = 1000
                num_batches = valid_cos_unselected_vectors.size(0) // batch_size + 1
                for i in trange(num_batches):
                    start = i * batch_size
                    end = min((i + 1) * batch_size, valid_cos_unselected_vectors.size(0))
                    batch_valid_cos_matrxi_unselected = valid_cos_unselected_vectors[start:end, :]
                    batch_cos_matrix = batch_valid_cos_matrxi_unselected @ valid_cos_unselected_vectors.T
                    batch_count = count_valid_rows(batch_cos_matrix, valid_cos_arr)
                    count[start:end] = batch_count
                k = min(2, count.shape[0])
                topk_values, topk_indices = torch.topk(count, k=k)
                current_avail_actions_indices = valid_cos_indices[topk_indices]
                
                # take action
                random_index = torch.randint(0, len(topk_indices), (1,))
                action = current_avail_actions_indices[random_index]
                board[action] = True
                legal_actions[action] = False
            selected_num = torch.sum(board)
            print(selected_num)
            np_board = board.cpu().numpy()
            np.save(save_dir / f"greedy_board_{selected_num}.npy", np_board)
