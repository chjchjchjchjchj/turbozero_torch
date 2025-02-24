from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import numpy as np
from core.env import Env, EnvConfig
from pathlib import Path

@dataclass
class VectorSelectionEnvConfig(EnvConfig):
    env_type: str = 'vector_selection'
    board_size: int = 196560
    dim: int = 24
    vector_path: str = '24D_196560_1.npy'
    save_dir: str = "./alphazero_vectors_selection"
    valid_cos_array: List[float] = (-1, -0.25, 0, 0.25)
    lower_bound: int = 488

class VectorSelectionEnv(Env):
    def __init__(self, 
        parallel_envs: int, 
        config: VectorSelectionEnvConfig, 
        device: torch.device, 
        debug: bool
    ) -> None:
        self.board_size = config.board_size
        self.dim = config.dim
        self.vector_path = config.vector_path
        self.all_vectors = torch.from_numpy(np.load(self.vector_path)).float().to(device)
        self.valid_cos_array = torch.tensor(config.valid_cos_array, device=device)
        self.lower_bound = config.lower_bound
        
        # 设置状态和动作空间
        state_shape = torch.Size((1, self.dim, self.dim))  # 一维状态空间
        policy_shape = torch.Size((self.board_size,))   # 动作空间大小
        value_shape = torch.Size((1,))                  # 值函数输出维度

        super().__init__(
            parallel_envs = parallel_envs,
            config = config,
            device = device,
            state_shape = state_shape,
            policy_shape = policy_shape,
            value_shape = value_shape,
            debug = debug,
            num_players = 1
        )

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化状态变量
        self.board = torch.zeros((self.parallel_envs, self.board_size), dtype=torch.int, device=device)
        self.legal_actions = torch.zeros((self.parallel_envs, self.board_size), dtype=torch.bool, device=device)

    def get_legal_actions(self):
        # 对于初始状态，所有动作都是合法的
        initial_env_idx = torch.all(self.board == 0, dim=1)
        self.legal_actions.zero_()
        self.legal_actions[initial_env_idx] = True

        # 对于非初始状态，使用向量化操作计算合法动作
        non_initial_idx = ~initial_env_idx
        if torch.any(non_initial_idx):
            # 获取非初始状态的环境索引
            non_initial_envs = torch.where(non_initial_idx)[0]
            
            # 获取每个环境中已选择和未选择的向量索引
            board_non_initial = self.board[non_initial_envs]
            res_mask = (board_non_initial == 0)
            selected_mask = (board_non_initial == 1)
            
            # 计算所有环境的余弦相似度矩阵
            rest_vectors = self.all_vectors.unsqueeze(0).expand(len(non_initial_envs), -1, -1)
            selected_vectors = self.all_vectors.unsqueeze(0).expand(len(non_initial_envs), -1, -1)
            
            # 使用掩码选择相关向量
            rest_vectors = rest_vectors * res_mask.unsqueeze(-1)
            selected_vectors = selected_vectors * selected_mask.unsqueeze(-1)
            
            # 计算余弦相似度
            try:
                cos_matrix = torch.bmm(rest_vectors, selected_vectors.transpose(1, 2))
            except Exception as e:
                import ipdb; ipdb.set_trace()
                print(f"rest_vectors: {rest_vectors.shape}, selected_vectors: {selected_vectors.shape}")
            
            # 检查每个余弦值是否有效
            valid_mask = torch.isclose(cos_matrix.unsqueeze(-1), 
                                     self.valid_cos_array.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                     atol=1e-8)
            
            # 确定合法动作
            valid_actions = torch.all(torch.any(valid_mask, dim=3), dim=2) & res_mask
            self.legal_actions[non_initial_envs] = valid_actions

        return self.legal_actions

    def is_row_valid(self, cos_matrix, valid_cos_array, tol=1e-8):
        result = torch.isclose(cos_matrix.unsqueeze(-1), 
                             valid_cos_array.unsqueeze(0).unsqueeze(0), 
                             atol=tol)
        valid_rows = torch.all(torch.any(result, dim=2), dim=1)
        return valid_rows

    def push_actions(self, actions):
        
        # 更新棋盘状态
        self.board[torch.arange(self.parallel_envs, device=self.device), actions] = 1
        # 更新状态表示
        # 计算已选择向量的自相关矩阵
        selected_vectors = self.all_vectors.unsqueeze(0).expand(self.parallel_envs, -1, -1)  # [batch, num_vectors, dim]
        selected_mask = self.board.unsqueeze(-1)  # [batch, num_vectors, 1]
        
        # 将未选择的向量置为0
        selected_vectors = selected_vectors * selected_mask  # [batch, num_vectors, dim]
        
        # 计算自相关矩阵
        self.states = torch.bmm(selected_vectors.transpose(1,2), selected_vectors)  # [batch, dim, dim]
    
    def next_turn(self):
        pass

    def reset(self, seed=None) -> int:
        if seed is not None:
            torch.manual_seed(seed)
        seed = 0
        self.states.zero_()
        self.terminated.zero_()
        self.board.zero_()
        self.legal_actions.zero_()
        return seed
        

    def is_terminal(self):
        # 当没有合法动作时终止
        return ~torch.any(self.get_legal_actions(), dim=1)

    def get_rewards(self, player_ids: Optional[torch.Tensor] = None):
        # 奖励是已选择向量数量与目标数量的比值
        selected_count = torch.sum(self.board, dim=1)
        rewards = selected_count / (self.lower_bound + 12)
        return rewards

    def save_node(self):
        return (self.board.clone(),)

    def load_node(self, load_envs: torch.Tensor, saved: Tuple[torch.Tensor]):
        self.board[load_envs] = saved[0][load_envs]
        self.states = self.build_states()
    
    def build_states(self):
        selected_vectors = self.all_vectors.unsqueeze(0).expand(self.parallel_envs, -1, -1)  # [batch, num_vectors, dim]
        selected_mask = self.board.unsqueeze(-1)  # [batch, num_vectors, 1]
        
        # 将未选择的向量置为0
        selected_vectors = selected_vectors * selected_mask  # [batch, num_vectors, dim]
        
        # 计算自相关矩阵
        self.states = torch.bmm(selected_vectors.transpose(1,2), selected_vectors)  # [batch, dim, dim]
        return self.states
    
    def print_state(self, last_action: Optional[int] = None) -> None:
        print(self.board)
        print(f"number of selected vectors: {torch.sum(self.board, dim=1)}")
