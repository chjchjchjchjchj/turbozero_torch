import torch
from core.train.trainer import Trainer, TrainerConfig
from core.utils.history import Metric, TrainingMetrics
from envs.vector_selection.collector import VectorSelectionCollector

class VectorSelectionTrainer(Trainer):
    def __init__(self,
        config: TrainerConfig,
        collector: VectorSelectionCollector,
        tester,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        raw_train_config: dict,
        raw_env_config: dict,
        history: TrainingMetrics,
        log_results: bool = True,
        interactive: bool = True,
        run_tag: str = 'vector_selection',
        debug: bool = False
    ):
        super().__init__(
            config=config,
            collector=collector,
            tester=tester,
            model=model,
            optimizer=optimizer,
            device=device,
            raw_train_config=raw_train_config,
            raw_env_config=raw_env_config,
            history=history,
            log_results=log_results,
            interactive=interactive,
            run_tag=run_tag,
            debug=debug
        )

        if self.history.cur_epoch == 0:
            self.history.episode_metrics.update({
                'reward': Metric(
                    name='reward',
                    xlabel='Episode',
                    ylabel='Reward',
                    addons={'running_mean': 100},
                    maximize=True,
                    alert_on_best=self.log_results
                )
            })

            self.history.epoch_metrics.update({
                'avg_reward': Metric(
                    name='avg_reward',
                    xlabel='Epoch',
                    ylabel='Average Reward',
                    maximize=True,
                    alert_on_best=self.log_results,
                    proper_name='Average Reward'
                )
            })
    
    def add_collection_metrics(self, episodes):
        for episode in episodes:
            reward = episode[-1][2].item()  # 获取最后一个状态的奖励
            self.history.add_episode_data({
                'reward': reward
            }, log=self.log_results)
            
    def add_epoch_metrics(self):
        episode_rewards = [m['reward'] for m in self.history.episode_metrics['reward'].data]
        if episode_rewards:
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            self.history.add_epoch_data({
                'avg_reward': avg_reward
            }, log=self.log_results) 