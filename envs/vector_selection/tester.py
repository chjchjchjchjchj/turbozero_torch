import torch
from core.test.tester import Tester, TesterConfig
from core.utils.history import TrainingMetrics
from typing import Optional

class VectorSelectionTester(Tester):
    def add_evaluation_metrics(self, episodes):
        for episode in episodes:
            reward = episode[-1][2].item()  # 获取最后一个状态的奖励
            self.history.add_evaluation_data({
                'reward': reward
            }, log=self.log_results)