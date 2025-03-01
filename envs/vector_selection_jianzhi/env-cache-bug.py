
from dataclasses import dataclass
from typing import Optional, Tuple
from colorama import Fore
import torch
from core.env import Env, EnvConfig
import numpy as np
from pathlib import Path
from tqdm import trange

def sorted_tensor_top_k_by_approx_count(tensor, d=0, n=0, tolerance=1e-2, top_k=1):
    is_close_to_05 = torch.abs(tensor - 1/4) <= tolerance
    counts = is_close_to_05.sum(dim=1)
    sorted_indices = torch.argsort(counts, descending=True)
    sorted_tensor = tensor[sorted_indices]
    sorted_counts = counts[sorted_indices]
    unique_sorted_tensor = torch.unique(sorted_counts)    

    top_k = np.minimum(top_k, unique_sorted_tensor.shape[0])

    top_k_threshold = unique_sorted_tensor[-top_k]

    top_k_tensor = sorted_tensor[sorted_counts >= top_k_threshold]
    top_k_indices = sorted_indices[sorted_counts >= top_k_threshold]
    return top_k_indices

def count_valid_nums(remain_cos_matrix):
    device = remain_cos_matrix.device
    return torch.sum(torch.isclose(remain_cos_matrix, torch.tensor(0.25, device=device, dtype=torch.float16)), dim=1) + torch.sum(torch.isclose(remain_cos_matrix, torch.tensor(-0.25, device=device, dtype=torch.float16)), dim=1) + torch.sum(torch.isclose(remain_cos_matrix, torch.tensor(0, device=device, dtype=torch.float16)), dim=1) + torch.sum(torch.isclose(remain_cos_matrix, torch.tensor(-1, device=device, dtype=torch.float16)), dim=1)

# def count_valid_nums_for_loop(valid_unselected_vectors, unselected_vectors):
#     device = valid_unselected_vectors.device
#     num_vectors = valid_unselected_vectors.shape[0]

#     # 初始化 count 统计数组
#     count_all = torch.zeros(num_vectors, dtype=torch.int32, device=device)

#     # 逐个计算余弦相似度并统计 count
#     for i in range(num_vectors):
#         count = 0
#         for j in trange(unselected_vectors.shape[0]):
#             dot_product = torch.dot(valid_unselected_vectors[i], unselected_vectors[j]).to(device=device, dtype=torch.float16)  # 计算点积
#             if torch.isclose(dot_product, torch.tensor(0.25, device=device, dtype=torch.float16)):
#                 count += 1
#             elif torch.isclose(dot_product, torch.tensor(-0.25, device=device, dtype=torch.float16)):
#                 count += 1
#             elif torch.isclose(dot_product, torch.tensor(0, device=device, dtype=torch.float16)):
#                 count += 1
#             elif torch.isclose(dot_product, torch.tensor(-1, device=device, dtype=torch.float16)):
#                 count += 1
#         count_all[i] = count  # 存入 count 统计数组

#     # 排序，获取最大 count 对应的索引
#     sorted_indices = torch.argsort(count_all, descending=True)

#     return sorted_indices

def count_valid_nums_batch_loop(valid_unselected_vectors, unselected_vectors, batch_size=8000):
    device = valid_unselected_vectors.device
    num_vectors = valid_unselected_vectors.shape[0]
    num_unselected = unselected_vectors.shape[0]

    # # 初始化 count 统计数组
    # count_all = torch.zeros(num_vectors, dtype=torch.int32, device=device)

    # # 遍历 valid_unselected_vectors
    # for i in trange(num_vectors):
    #     count = 0
    #     dot_products = torch.matmul(valid_unselected_vectors[i].unsqueeze(0), unselected_vectors.T).squeeze(0).to(dtype=torch.float16)
    #     count += (
    #         torch.sum(torch.isclose(dot_products, torch.tensor(0.25, dtype=torch.float16))) +
    #         torch.sum(torch.isclose(dot_products, torch.tensor(-0.25, dtype=torch.float16))) +
    #         torch.sum(torch.isclose(dot_products, torch.tensor(0, dtype=torch.float16))) +
    #         torch.sum(torch.isclose(dot_products, torch.tensor(-1, dtype=torch.float16)))
    #     )
        
    #     count_all[i] = count  # 存入 count 统计数组

        # 初始化 count 统计数组
    count_all = torch.zeros(num_vectors, dtype=torch.int32, device=device)

    # 分块计算矩阵乘法
    num_batches = (num_vectors + batch_size - 1) // batch_size

    for i in trange(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_vectors)

        # 计算分块的余弦相似度矩阵
        remain_cos_matrix = valid_unselected_vectors[start:end] @ unselected_vectors.T  # (batch_size, 196560)
        remain_cos_matrix = remain_cos_matrix.to(dtype=torch.float16)

        # 统计 count（满足条件的个数）
        count_batch = (
            torch.sum(torch.isclose(remain_cos_matrix, torch.tensor(0.25, device=device, dtype=torch.float16)), dim=1) +
            torch.sum(torch.isclose(remain_cos_matrix, torch.tensor(-0.25, device=device, dtype=torch.float16)), dim=1) +
            torch.sum(torch.isclose(remain_cos_matrix, torch.tensor(0, device=device, dtype=torch.float16)), dim=1) +
            torch.sum(torch.isclose(remain_cos_matrix, torch.tensor(-1, device=device, dtype=torch.float16)), dim=1)
        )

        # 存入 count_all
        count_all[start:end] = count_batch

    # 排序，获取最大 count 对应的索引
    sorted_indices = torch.argsort(count_all, descending=True)

    return sorted_indices, count_all



@dataclass
class VectorSelectionJianzhiEnvConfig(EnvConfig):
    board_size: int = 196560
    dim: int = 24
    lower_bound: int = 500
    valid_cos_arr: torch.tensor = torch.tensor([-1, -1/4, 1/4, 0, 1], dtype=torch.float16)
    save_path: str = "/data/haojun/max_board"
    all_vectors_path: str = "24D_196560_1.npy"
    FloatType: str = "torch.float16"
    

class VectorSelectionJianzhiEnv(Env):
    def __init__(self,
        parallel_envs: int,
        config: VectorSelectionJianzhiEnvConfig, 
        device: torch.device,
        debug=False
    ) -> None:
        # 添加dtype转换逻辑
        dtype_map = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
        }
        self.FloatType = dtype_map[config.FloatType]
        self.config = config
        self.lower_bound = config.lower_bound
        self.board_size = config.board_size
        self.dim = config.dim
        self.valid_cos_arr = torch.tensor(config.valid_cos_arr, dtype=self.FloatType, device=device, requires_grad=False)
        super().__init__(
            parallel_envs=parallel_envs, 
            config=config, 
            device=device,
            num_players=1,
            state_shape=torch.Size([1, self.lower_bound, self.lower_bound]),
            policy_shape=torch.Size([self.board_size]),
            value_shape=torch.Size([1]),
            debug=debug
        )
        self.all_vectors = torch.from_numpy(np.load(config.all_vectors_path)).to(device=device, dtype=self.FloatType)
        self.states = torch.zeros((self.parallel_envs, 1, self.lower_bound, self.lower_bound), dtype=self.FloatType, device=device, requires_grad=False)
        self.boards = torch.zeros((self.parallel_envs, self.board_size), dtype=torch.bool, device=device, requires_grad=False)
        self.rewards = torch.zeros((self.parallel_envs, ), dtype=self.FloatType, device=device, requires_grad=False)

        self.save_path = Path(config.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.cached_legal_actions = torch.ones((self.parallel_envs, self.board_size), dtype=torch.bool, device=device)

    def reset(self, seed=None) -> int:
        if seed is not None:
            torch.manual_seed(seed)
        else:
            seed = 0
        self.max_n_vectors = 0
        self.states.zero_()
        self.boards.zero_()
        self.terminated.zero_()
        self.cached_legal_actions.fill_(True)
        return seed

    def reset_terminated_states(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            torch.manual_seed(seed)
        else:
            seed = 0
        self.states *= torch.logical_not(self.terminated).view(self.parallel_envs, 1, 1, 1)
        self.boards *= torch.logical_not(self.terminated).view(self.parallel_envs, 1)
        self.cached_legal_actions[self.terminated] = True
        self.terminated.zero_()
        return seed

    def next_turn(self):
        pass

    def get_rewards(self, player_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.rewards

    def update_terminated(self) -> None:
        self.terminated = self.is_terminal()

    def is_terminal(self):
        return (self.get_legal_actions().sum(dim=1, keepdim=True) == 0).flatten()
    
    # def get_legal_actions(self) -> torch.Tensor:
    #     return self.cached_legal_actions

    def get_legal_actions(self) -> torch.Tensor:
        legal_actions = self.cached_legal_actions.clone()
        for env_idx in range(self.parallel_envs):
            valid_unselected_indices = torch.where(legal_actions[env_idx])[0]
            valid_unselected_vectors = self.all_vectors[valid_unselected_indices]
            sorted_indices, count = count_valid_nums_batch_loop(valid_unselected_vectors, valid_unselected_vectors)
            sorted_indices = sorted_indices[10:]
            min_valid_indices = valid_unselected_indices[sorted_indices]
            legal_actions[env_idx, min_valid_indices] = False
        return legal_actions
            


    #     legal_actions = torch.zeros((self.parallel_envs, self.board_size), dtype=torch.bool, device=self.device, requires_grad=False)
        
    #     # 计算每个GPU应处理的环境数量
    #     n_gpus = torch.cuda.device_count()
    #     envs_per_gpu = (self.parallel_envs + n_gpus - 1) // n_gpus  # 向上取整
        
    #     for gpu_idx in range(n_gpus):
    #         start_idx = gpu_idx * envs_per_gpu
    #         end_idx = min((gpu_idx + 1) * envs_per_gpu, self.parallel_envs)
            
    #         if start_idx >= self.parallel_envs:
    #             break
                
    #         # 将数据移动到对应的GPU
    #         device = torch.device(f'cuda:{gpu_idx}')
    #         current_boards = self.boards[start_idx:end_idx].to(device)
    #         current_all_vectors = self.all_vectors.to(device)
    #         current_valid_cos_arr = self.valid_cos_arr.to(device)
            
    #         # 在当前GPU上处理一批环境
    #         for env_idx in range(start_idx, end_idx):
    #             local_idx = env_idx - start_idx
    #             board = current_boards[local_idx]
    #             selected_vectors = current_all_vectors[board]
                
    #             if len(selected_vectors) == 0:
    #                 legal_actions[env_idx] = True
    #                 continue
                    
    #             unselected_vectors = current_all_vectors[~board]
    #             unselected_indices = torch.where(~board)[0]
    #             cos_matrix = unselected_vectors @ selected_vectors.T
    #             valid_rows = self.find_valid_rows(cos_matrix, current_valid_cos_arr)
    #             valid_rows = unselected_indices[valid_rows]
    #             legal_actions[env_idx, valid_rows] = True
                
    #         # 清理当前GPU内存
    #         torch.cuda.empty_cache()
            
    #     return legal_actions
    
    # def find_valid_rows(self, cos_matrix, valid_cos_arr, tolerance=1e-5):
    #     # 将valid_cos_arr扩展为与cos_matrix匹配的形状，以便进行广播比较
    #     expanded_valid = valid_cos_arr.unsqueeze(0).expand(cos_matrix.shape[0], -1)
        
    #     # 对每个元素，检查是否与valid_cos_arr中的任何值接近
    #     # 使用外积比较，得到shape为(cos_matrix.shape[0], cos_matrix.shape[1], valid_cos_arr.shape[0])的布尔张量
    #     is_close = torch.abs(cos_matrix.unsqueeze(-1) - valid_cos_arr) < tolerance
        
    #     # 对最后一个维度取或运算，检查每个元素是否至少与一个valid值接近
    #     has_valid_value = torch.any(is_close, dim=-1)
        
    #     # 对第二个维度取与运算，确保行中的所有元素都是有效的
    #     valid_rows = torch.all(has_valid_value, dim=1)
        
    #     return valid_rows

    def find_valid_rows(self, cos_matrix, valid_cos_arr, tolerance=1e-5):
        # 使用广播和向量化操作
        diff = torch.abs(cos_matrix.unsqueeze(-1) - valid_cos_arr)
        is_valid = torch.any(diff < tolerance, dim=-1)
        return torch.all(is_valid, dim=-1)


    def push_actions(self, actions):
        # # generate cached_legal_actions
        # self.boards[torch.arange(self.parallel_envs), actions] = True
        # self.cached_legal_actions.fill_(False)
        # for env_idx in range(self.parallel_envs):
        #     board = self.boards[env_idx]
        #     selected_vectors = self.all_vectors[board]
        #     if len(selected_vectors) == 0:
        #         self.cached_legal_actions[env_idx] = True
        #         continue
        #     unselected_vectors = self.all_vectors[~board]
        #     unselected_indices = torch.where(~board)[0]
        #     cos_matrix = unselected_vectors @ selected_vectors.T
        #     valid_rows = self.find_valid_rows(cos_matrix, self.valid_cos_arr)
        #     valid_rows = unselected_indices[valid_rows]
        #     self.cached_legal_actions[env_idx, valid_rows] = True
        
        # # generate cached_legal_actions
        # self.boards[torch.arange(self.parallel_envs), actions] = True
        # self.cached_legal_actions[torch.arange(self.parallel_envs), actions] = False
        # for env_idx in range(self.parallel_envs):
        #     board = self.boards[env_idx]
        #     selected_vectors = self.all_vectors[board]
        #     if len(selected_vectors) == 0:
        #         self.cached_legal_actions[env_idx] = True
        #         continue
        #     unselected_vectors = self.all_vectors[self.cached_legal_actions[env_idx]]
        #     unselected_indices = torch.where(self.cached_legal_actions[env_idx])[0]
        #     cos_matrix = unselected_vectors @ selected_vectors.T
        #     valid_rows = self.find_valid_rows(cos_matrix, self.valid_cos_arr)
        #     valid_unselected_vectors = unselected_vectors[valid_rows]
        #     valid_unselected_indices = unselected_indices[valid_rows]
        
        #     # remain_cos_matrix = valid_unselected_vectors @ unselected_vectors.T

        #     # 计算索引
        #     # sorted_indices = count_valid_nums_for_loop(valid_unselected_vectors, unselected_vectors)
        #     sorted_indices, count = count_valid_nums_batch_loop(valid_unselected_vectors, unselected_vectors)
        #     sorted_indices = sorted_indices[10:]
        #     min_valid_indices = valid_unselected_indices[sorted_indices]
        #     print(torch.all(self.find_valid_rows(self.all_vectors[valid_unselected_indices[sorted_indices]] @ selected_vectors.T, self.valid_cos_arr)))
        #     if not torch.all(self.find_valid_rows(self.all_vectors[valid_unselected_indices[sorted_indices]] @ selected_vectors.T, self.valid_cos_arr)):
        #         import ipdb; ipdb.set_trace()
        #         print(f"env_idx: {env_idx}, sorted_indices: {sorted_indices}, min_valid_indices: {min_valid_indices}")
        #     self.cached_legal_actions[env_idx, min_valid_indices] = False
        
        # generate cached_legal_actions
        self.boards[torch.arange(self.parallel_envs), actions] = True
        self.cached_legal_actions[torch.arange(self.parallel_envs), actions] = False
        for env_idx in range(self.parallel_envs):
            board = self.boards[env_idx]
            selected_vectors = self.all_vectors[board]
            if len(selected_vectors) == 0:
                self.cached_legal_actions[env_idx] = True
                continue
            unselected_vectors = self.all_vectors[self.cached_legal_actions[env_idx]]
            unselected_indices = torch.where(self.cached_legal_actions[env_idx])[0]
            cos_matrix = unselected_vectors @ selected_vectors.T
            valid_rows = self.find_valid_rows(cos_matrix, self.valid_cos_arr)
            invalid_rows = unselected_indices[~valid_rows]
            self.cached_legal_actions[env_idx, invalid_rows] = False

            valid_unselected_indices = unselected_indices[valid_rows]
            valid_unselected_vectors = unselected_vectors[valid_rows]
        
            # remain_cos_matrix = valid_unselected_vectors @ unselected_vectors.T

            # 计算索引
            # sorted_indices = count_valid_nums_for_loop(valid_unselected_vectors, unselected_vectors)
            sorted_indices, count = count_valid_nums_batch_loop(valid_unselected_vectors, unselected_vectors)
            sorted_indices = sorted_indices[:10]
            min_valid_indices = valid_unselected_indices[sorted_indices]

            print(torch.all(self.find_valid_rows(self.all_vectors[valid_unselected_indices[sorted_indices]] @ selected_vectors.T, self.valid_cos_arr)))
            if not torch.all(self.find_valid_rows(self.all_vectors[valid_unselected_indices[sorted_indices]] @ selected_vectors.T, self.valid_cos_arr)):
                print(f"env_idx: {env_idx}, sorted_indices: {sorted_indices}, min_valid_indices: {min_valid_indices}")
            self.cached_legal_actions[env_idx, min_valid_indices] = False

        # generate states
        for env_idx in range(self.parallel_envs):
            board = self.boards[env_idx]
            selected_vectors = self.all_vectors[board]
            # 计算已选向量之间的余弦相似度矩阵
            cos_matrix = selected_vectors @ selected_vectors.T
            # 直接填充子矩阵
            self.states[env_idx, 0, :cos_matrix.shape[0], :cos_matrix.shape[0]] = cos_matrix

        
        # save states
        board = self.boards.cpu().numpy()
        current_max = np.sum(board, axis=1)
        arg_max_env_idx = np.argmax(current_max)
        num = current_max[arg_max_env_idx]
        if num > self.max_n_vectors:
            self.max_n_vectors = num
            max_board = board[arg_max_env_idx]
            save_path = self.save_path / f"max_board_{num}.npy"
            np.save(save_path, max_board)
            print(f"save max_board to {save_path}")

    # def push_actions(self, actions):

    #     # 使用批处理计算legal actions
    #     batch_size = 1024  # 可调整的批量大小
    #     self.boards[torch.arange(self.parallel_envs), actions] = True
    #     self.cached_legal_actions.fill_(False)
        
    #     for batch_start in range(0, self.parallel_envs, batch_size):
    #         batch_end = min(batch_start + batch_size, self.parallel_envs)
    #         batch_boards = self.boards[batch_start:batch_end]
    #         batch_selected = self.all_vectors[batch_boards]
            
    #         if len(batch_selected) == 0:
    #             self.cached_legal_actions[batch_start:batch_end] = True
    #             continue
                
    #         batch_unselected = self.all_vectors[~batch_boards]
    #         batch_indices = torch.where(~batch_boards)[0]
            
    #         # 使用矩阵乘法计算余弦相似度
    #         cos_matrix = torch.matmul(batch_unselected, batch_selected.transpose(-2, -1))
    #         valid_rows = self.find_valid_rows(cos_matrix, self.valid_cos_arr)
    #         valid_indices = batch_indices[valid_rows]
            
    #         self.cached_legal_actions[batch_start:batch_end, valid_indices] = True

    #     # generate states
    #     for env_idx in range(self.parallel_envs):
    #         board = self.boards[env_idx]
    #         selected_vectors = self.all_vectors[board]
    #         # 计算已选向量之间的余弦相似度矩阵
    #         cos_matrix = selected_vectors @ selected_vectors.T
    #         # 直接填充子矩阵
    #         self.states[env_idx, 0, :cos_matrix.shape[0], :cos_matrix.shape[0]] = cos_matrix

        
    #     # save states
    #     board = self.boards.cpu().numpy()
    #     current_max = np.sum(board, axis=1)
    #     arg_max_env_idx = np.argmax(current_max)
    #     num = current_max[arg_max_env_idx]
    #     if num > self.max_n_vectors:
    #         self.max_n_vectors = num
    #         max_board = board[arg_max_env_idx]
    #         save_path = self.save_path / f"max_board_{num}.npy"
    #         np.save(save_path, max_board)
    #         # print(f"save max_board to {save_path}")
        
    def save_node(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states.clone(), self.boards.clone(), self.cached_legal_actions.clone()

    def load_node(self, load_envs: torch.Tensor, saved: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        saved_states, saved_boards, saved_cached_legal_actions = saved
        load_envs_expnd_states = load_envs.view(self.parallel_envs, 1, 1, 1)
        load_envs_expnd_boards = load_envs.view(self.parallel_envs, 1)
        load_envs_expnd_cached_legal_actions = load_envs.view(self.parallel_envs, 1)
        self.states = saved_states.clone() * load_envs_expnd_states + self.states * (~load_envs_expnd_states)
        self.boards = saved_boards.clone() * load_envs_expnd_boards + self.boards * (~load_envs_expnd_boards)
        self.cached_legal_actions = saved_cached_legal_actions.clone() * load_envs_expnd_cached_legal_actions + self.cached_legal_actions * (~load_envs_expnd_cached_legal_actions)
        self.update_terminated()

    
        
        
