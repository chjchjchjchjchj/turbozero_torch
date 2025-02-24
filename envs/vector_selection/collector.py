import torch
from core.algorithms.evaluator import TrainableEvaluator
from core.train.collector import Collector
from envs.vector_selection.env import VectorSelectionEnvConfig

class VectorSelectionCollector(Collector):
    def __init__(self,
        evaluator: TrainableEvaluator,
        episode_memory_device: torch.device
    ) -> None:
        super().__init__(evaluator, episode_memory_device)
        assert isinstance(evaluator.env.config, VectorSelectionEnvConfig)
    
    def assign_rewards(self, terminated_episodes, terminated):
        episodes = []
        if terminated.any():
            term_indices = terminated.nonzero(as_tuple=False).flatten()
            rewards = self.evaluator.env.get_rewards().clone().cpu().numpy()
            for i, episode in enumerate(terminated_episodes):
                episode_with_rewards = []
                ti = term_indices[i]
                reward = rewards[ti]
                for (inputs, visits, legal_actions) in episode:
                    if visits.sum():  # 只添加有效移动的状态
                        episode_with_rewards.append((inputs, visits, torch.tensor(reward, dtype=torch.float32, requires_grad=False, device=inputs.device), legal_actions))
                episodes.append(episode_with_rewards)
        return episodes 