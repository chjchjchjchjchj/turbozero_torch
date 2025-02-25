
from dataclasses import dataclass
from typing import Optional, Tuple
from colorama import Fore
import torch
from core.env import Env, EnvConfig
import numpy as np
from pathlib import Path

@dataclass
class VectorSelectionEnvConfig(EnvConfig):
    board_size: int = 196560
    dim: int = 24
    lower_bound: int = 500
    valid_cos_arr: torch.tensor = torch.tensor([-1, -1/4, 1/4, 0, 1], dtype=torch.float16)
    save_path: str = "/data/haojun/max_board"
    all_vectors_path: str = "24D_196560_1.npy"
    FloatType: str = "torch.float16"
    

class VectorSelectionEnv(Env):
    def __init__(self,
        parallel_envs: int,
        config: VectorSelectionEnvConfig, 
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

    def reset(self, seed=None) -> int:
        if seed is not None:
            torch.manual_seed(seed)
        else:
            seed = 0
        self.max_n_vectors = 0
        self.states.zero_()
        self.boards.zero_()
        self.terminated.zero_()
        return seed

    def reset_terminated_states(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            torch.manual_seed(seed)
        else:
            seed = 0
        self.states *= torch.logical_not(self.terminated).view(self.parallel_envs, 1, 1, 1)
        self.boards *= torch.logical_not(self.terminated).view(self.parallel_envs, 1)
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

    def get_legal_actions(self) -> torch.Tensor:
        legal_actions = torch.zeros((self.parallel_envs, self.board_size), dtype=torch.bool, device=self.device, requires_grad=False)
        for env_idx in range(self.parallel_envs):
            board = self.boards[env_idx]
            selected_vectors = self.all_vectors[board]
            if len(selected_vectors) == 0:
                legal_actions[env_idx] = True
                continue
            unselected_vectors = self.all_vectors[~board]
            unselected_indices = torch.where(~board)[0]
            cos_matrix = unselected_vectors @ selected_vectors.T
            valid_rows = self.find_valid_rows(cos_matrix, self.valid_cos_arr)
            try:
                valid_rows = unselected_indices[valid_rows]
                legal_actions[env_idx, valid_rows] = True
            except Exception as e:
                import ipdb; ipdb.set_trace()
                print("")
        return legal_actions
    
    def find_valid_rows(self, cos_matrix, valid_cos_arr, tolerance=1e-5):
        # 将valid_cos_arr扩展为与cos_matrix匹配的形状，以便进行广播比较
        expanded_valid = valid_cos_arr.unsqueeze(0).expand(cos_matrix.shape[0], -1)
        
        # 对每个元素，检查是否与valid_cos_arr中的任何值接近
        # 使用外积比较，得到shape为(cos_matrix.shape[0], cos_matrix.shape[1], valid_cos_arr.shape[0])的布尔张量
        is_close = torch.abs(cos_matrix.unsqueeze(-1) - valid_cos_arr) < tolerance
        
        # 对最后一个维度取或运算，检查每个元素是否至少与一个valid值接近
        has_valid_value = torch.any(is_close, dim=-1)
        
        # 对第二个维度取与运算，确保行中的所有元素都是有效的
        valid_rows = torch.all(has_valid_value, dim=1)
        
        return valid_rows

    def push_actions(self, actions):
        self.boards[torch.arange(self.parallel_envs), actions] = True

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
        
    def save_node(self) -> torch.Tensor:
        return self.states.clone()

    def load_node(self, load_envs: torch.Tensor, saved: torch.Tensor):
        load_envs_expnd = load_envs.view(self.parallel_envs, 1, 1, 1)
        self.states = saved.clone() * load_envs_expnd + self.states * (~load_envs_expnd)
        self.update_terminated()

    
        
        
