from dataclasses import dataclass
from typing import Optional, Tuple
from colorama import Fore
import torch
from core.env import Env, EnvConfig
import numpy as np
import zarr
from pathlib import Path
from tqdm import trange
import time

@dataclass
class VectorSelectionCosEnvConfig(EnvConfig):
    board_size: int = 196560
    dim: int = 24
    cos_matrix_zarr_dir: str = "/data/haojun/cos_matrix"
    lower_bound: int = 500
    valid_cos_arr: np.array = np.array([-1, -1/4, 1/4, 0, 1], dtype=np.float16)

    all_vectors_path: str = "24D_196560_1.npy"
    save_path: str = "/data/haojun/max_board"

class VectorSelectionCosEnv(Env):
    def __init__(self,
        parallel_envs: int,
        config: VectorSelectionCosEnvConfig, 
        device: torch.device,
        debug=False,
        adj: torch.Tensor=None
    ) -> None:
        self.config = config
        self.lower_bound = config.lower_bound
        self.cos_matrix_zarr_dir = Path(config.cos_matrix_zarr_dir)
        self.board_size = config.board_size
        self.dim = config.dim
        self.valid_cos_arr = torch.tensor(config.valid_cos_arr, dtype=torch.float16, device=device, requires_grad=False)
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
        self.board = torch.zeros((self.parallel_envs, self.board_size), dtype=torch.bool, device=device, requires_grad=False)
        self.rewards = torch.zeros((self.parallel_envs, ), dtype=torch.float32, device=device, requires_grad=False)

        self.all_vectors_path = Path(self.config.all_vectors_path)
        # self.adj_matrix = self.get_adj_matrix()
        self.adj_matrix = adj

        self.save_path = Path(self.config.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.max_n_vectors = 0

        # 添加缓存的legal_actions
        self.cached_legal_actions = torch.ones((self.parallel_envs, self.board_size), dtype=torch.bool, device=device)

    
    

    # def get_adj_matrix(self) -> torch.Tensor:
    #     batch_size = 1000
    #     vectors = torch.from_numpy(np.load(self.all_vectors_path)).to(self.device).to(torch.float16)
    #     adj_matrix = []
    #     for i in trange(0, self.board_size, batch_size, desc="Generating Adjacency Matrix"):
    #         end = min(i + batch_size, self.board_size)
    #         vectors_a = vectors[i:end]
    #         cos_arr = vectors_a @ vectors.T
    #         m = cos_arr.shape[0]
    #         adj_arr = torch.min(torch.abs(cos_arr[:, :, None] - self.valid_cos_arr), axis=2)[0] < 1e-6
    #         adj_arr = adj_arr.cpu()
    #         adj_matrix.append(adj_arr)
    #     adj_matrix = torch.cat(adj_matrix, dim=0)
    #     return adj_matrix

    def reset(self, seed=None) -> int:
        if seed is not None:
            torch.manual_seed(seed)
        else:
            seed = 0
        self.states.zero_()
        self.board.zero_()
        self.terminated.zero_()

        # 重置legal_actions缓存
        self.cached_legal_actions.fill_(True)
        return seed

    def reset_terminated_states(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            torch.manual_seed()
        else:
            seed = 0
        self.states *= torch.logical_not(self.terminated).view(self.parallel_envs, 1, 1, 1)
        self.terminated.zero_() # TODO: check if this is necessary
        return seed

    def next_turn(self):
        pass

    def get_rewards(self, player_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # TODO: handle rewards in env instead of collector postprocessing
        return self.rewards
    
    def update_terminated(self) -> None:
        self.terminated = self.is_terminal()
    
    def is_terminal(self):
        return (self.get_legal_actions().sum(dim=1, keepdim=True) == 0).flatten()
    
    # def get_legal_actions(self) -> torch.Tensor:
    #     # legal_actions = torch.zeros((self.parallel_envs, self.board_size), dtype=torch.bool, device=self.device, requires_grad=False)
    #     # init_env_indices = ~torch.any(self.board, dim=1)
    #     # legal_actions[init_env_indices, ...] = True
    #     # non_init_env_indices = ~init_env_indices
    #     # if non_init_env_indices.sum() > 0:
    #     #     expanded_board = self.board[non_init_env_indices].view(non_init_env_indices.sum(), 1, 1, self.board_size)
    #     board = self.board.cpu()
    #     expanded_adj = self.adj_matrix.unsqueeze(0).expand(self.parallel_envs, -1, -1, -1)
    #     adjacent = torch.any(expanded_adj & board[:, None, :], dim=-1)
    #     legal_actions = adjacent & ~board
    #     print(legal_actions)

    #     # for i in range(self.parallel_envs):
    #     #     selected_indices = self.board[i].nonzero().flatten()
    #     #     if len(selected_indices) == 0:
    #     #         legal_actions[i] = torch.ones((self.board_size, ), dtype=torch.bool, device=self.device, requires_grad=False)
    #     #         continue
    #     #     for j in trange(self.board_size):
    #     #         cos_arr = zarr.load(self.cos_matrix_zarr_dir / f"{j}.zarr")
    #     #         cos_arr = torch.from_numpy(cos_arr).to(self.device)
    #     #         cos_arr = cos_arr[selected_indices]
    #     #         legal_actions[i, j] = torch.all(torch.min(torch.abs(cos_arr[:, None] - self.valid_cos_arr), axis=1)[0] < 1e-6)
    #     return legal_actions

    # def get_legal_actions(self) -> torch.Tensor:
    #     # 初始化结果张量
    #     legal_actions = torch.zeros((self.parallel_envs, self.board_size), dtype=torch.bool, device=self.device)
        
    #     # 使用批处理来处理大矩阵
    #     batch_size = 1000  # 可以根据可用内存调整这个值
    #     for start_idx in trange(0, self.board_size, batch_size, desc="Generating Legal Actions"):
    #         end_idx = min(start_idx + batch_size, self.board_size)
    #         # 只取部分邻接矩阵
    #         adj_batch = self.adj_matrix[start_idx:end_idx].to(self.device)
            
    #         # 对每个环境分别处理
    #         for env_idx in range(self.parallel_envs):
    #             # 找出当前环境中已选择的点
    #             selected = self.board[env_idx]
    #             if selected.sum() == 0:
    #                 legal_actions[env_idx, start_idx:end_idx] = True
    #                 continue
                
    #             # 计算这批节点是否与已选节点相邻
    #             is_adjacent = torch.any(adj_batch & selected, dim=1)
                
    #             # 更新结果
    #             legal_actions[env_idx, start_idx:end_idx] = is_adjacent & ~selected[start_idx:end_idx]
        
    #     print(f"legal_actions calculated wrong: {torch.any(torch.logical_and(self.board, legal_actions))}")
        
        
    #     return legal_actions.to(self.device)

    def get_legal_actions(self) -> torch.Tensor:
        return self.cached_legal_actions

    # def get_legal_actions(self) -> torch.Tensor:
    #     # 初始化结果张量
    #     print("get_legal_actions...")
    #     print(torch.where(self.board == True))
    #     # import ipdb; ipdb.set_trace()
    #     legal_actions = torch.zeros((self.parallel_envs, self.board_size), dtype=torch.bool, device=self.device)
    #     start_time = time.perf_counter()
        
    #     # 将board_size分配到多个GPU上
    #     num_gpus = torch.cuda.device_count()
    #     chunk_size = self.board_size // num_gpus
        
    #     # 为每个GPU创建CUDA流和存储结果的列表
    #     streams = [torch.cuda.Stream(device=f'cuda:{i}') for i in range(num_gpus)]
    #     gpu_results = []
        
    #     # 在每个GPU上启动异步计算
    #     for gpu_id in range(num_gpus):
    #         start_idx = gpu_id * chunk_size
    #         end_idx = start_idx + chunk_size if gpu_id < num_gpus - 1 else self.board_size
    #         gpu_device = torch.device(f'cuda:{gpu_id}')
            
    #         with torch.cuda.stream(streams[gpu_id]):
    #             # 异步将数据移动到GPU
    #             adj_batch = self.adj_matrix[start_idx:end_idx].to(gpu_device, non_blocking=True)
    #             board_gpu = self.board.to(gpu_device, non_blocking=True)
                
    #             gpu_legal_actions = torch.zeros((self.parallel_envs, end_idx - start_idx), 
    #                                         dtype=torch.bool, device=gpu_device)
                
    #             # 计算这个GPU负责的部分
    #             for env_idx in range(self.parallel_envs):
    #                 selected = board_gpu[env_idx]
    #                 if selected.sum() == 0:
    #                     gpu_legal_actions[env_idx] = True
    #                     continue
                    
    #                 is_adjacent = torch.any(adj_batch & selected, dim=1)
    #                 gpu_legal_actions[env_idx] = is_adjacent & ~selected[start_idx:end_idx]
                
    #             gpu_results.append((start_idx, end_idx, gpu_legal_actions))
        
    #     # 同步所有GPU
    #     torch.cuda.synchronize()
        
    #     # 收集所有GPU的结果
    #     for start_idx, end_idx, gpu_result in gpu_results:
    #         legal_actions[:, start_idx:end_idx] = gpu_result.to(self.device)
    #     end_time = time.perf_counter()

    #     print(f"legal_actions: {torch.where(legal_actions == True)}")
    #     print(f"{end_time - start_time:.2f} for get_legal_actions")
        
    #     return legal_actions

    # def find_non_adjacent_nodes(self, board, adj):
    #     """
    #     找到每个图中不与已选择的节点相连的所有节点
    #     :param board: 形状为 (parallel_env, N)，表示每个图中已选节点 (1) 和未选节点 (0)
    #     :param adj: 形状为 (N, N)，邻接矩阵
    #     :return: 一个形状为 (parallel_env, N) 的布尔矩阵，表示不与已选节点相连的所有节点
    #     """
        
    #     # 计算每个图中与已选节点相连的节点
    #     connected_nodes = (board.to(torch.int8) @ adj.to(torch.int8)) > 0  # (parallel_env, N)
        
    #     # 计算不与已选节点相连的所有节点
    #     non_adjacent_nodes = ~(connected_nodes | board)  # 取补集，同时排除已选节点
        
    #     return non_adjacent_nodes

    def find_non_adjacent_nodes(self, board, adj):
        """
        找到每个图中不与已选择的节点相连的所有节点
        :param board: 形状为 (parallel_env, N)，表示每个图中已选节点 (1) 和未选节点 (0)
        :param adj: 形状为 (N, N)，邻接矩阵
        :return: 一个形状为 (parallel_env, N) 的布尔矩阵，表示不与已选节点相连的所有节点
        """
        batch_size = 1000  # 可以根据显存大小调整这个值
        N = adj.shape[0]
        device = board.device
        
        # 初始化结果tensor
        connected_nodes = torch.zeros_like(board, dtype=torch.bool, device=device)
        
        # 分批处理邻接矩阵
        for start_idx in trange(0, N, batch_size, desc="find_non_adjacent_nodes"):
            end_idx = min(start_idx + batch_size, N)
            # 只加载部分邻接矩阵到GPU
            adj_batch = adj[start_idx:end_idx].to(device, dtype=torch.float32)
            
            # 计算部分结果
            batch_result = (board.to(torch.torch.float32) @ adj_batch.T) > 0
            connected_nodes[:, start_idx:end_idx] = batch_result
            
            # 释放显存
            del adj_batch
            torch.cuda.empty_cache()
        
        # 计算不与已选节点相连的所有节点
        non_adjacent_nodes = ~(connected_nodes | board)
        
        return non_adjacent_nodes
    
    def push_actions(self, actions):
        # print(f"actions: {actions} in push_actions")
        # print(f"self.board: {self.board[torch.arange(self.parallel_envs), actions]}")
        # print(f"board 中 True 的坐标: {torch.where(self.board == True)}")
        # import ipdb; ipdb.set_trace()
        self.board[torch.arange(self.parallel_envs), actions] = True
        # print(f"action 作用后, board 中 True 的坐标: {torch.where(self.board == True)}")

        # 更新legal_actions缓存
        batch_size = 1000
        # for start_idx in range(0, self.board_size, batch_size):
        #     end_idx = min(start_idx + batch_size, self.board_size)
        #     adj_batch = self.adj_matrix[start_idx:end_idx].to(self.device)
            
        #     for env_idx in range(self.parallel_envs):
        #         selected = self.board[env_idx]
        #         if selected.sum() == 0:
        #             self.cached_legal_actions[env_idx, start_idx:end_idx] = True
        #             continue
                
        #         is_adjacent = torch.any(adj_batch & selected, dim=1)
        #         self.cached_legal_actions[env_idx, start_idx:end_idx] = is_adjacent & ~selected[start_idx:end_idx]

        self.cached_legal_actions = self.find_non_adjacent_nodes(board=self.board, adj=self.adj_matrix)

        # generate states
        for i in trange(self.parallel_envs, desc="Generating States"):
            cos_matrix = []
            selected_indices = self.board[i].nonzero().flatten()
            if len(selected_indices) > 0:
                for selected_index in selected_indices:
                    cos_arr = zarr.load(self.cos_matrix_zarr_dir / f"{selected_index}.zarr")
                    cos_arr = torch.from_numpy(cos_arr).to(self.device)
                    cos_matrix.append(cos_arr[selected_indices])
                cos_matrix = torch.stack(cos_matrix)
                self.states[i, 0, :len(selected_indices), :len(selected_indices)] = cos_matrix
            else:
                self.states[i] = torch.zeros((1, self.lower_bound, self.lower_bound), dtype=torch.float32, device=self.device, requires_grad=False)
        
        # save states
        board = self.board.cpu().numpy()
        current_max = np.sum(board, axis=1)
        arg_max_env_idx = np.argmax(current_max)
        num = current_max[arg_max_env_idx]
        if num > self.max_n_vectors:
            self.max_n_vectors = num
            max_board = board[arg_max_env_idx]
            save_path = self.save_path / f"max_board_{num}.npy"
            np.save(save_path, max_board)
            print(f"save max_board to {save_path}")

    def save_node(self) -> torch.Tensor:
        return self.states.clone()

    def load_node(self, load_envs: torch.Tensor, saved: torch.Tensor):
        load_envs_expnd = load_envs.view(self.parallel_envs, 1, 1, 1)
        self.states = saved.clone() * load_envs_expnd + self.states * (~load_envs_expnd)
        self.update_terminated()
