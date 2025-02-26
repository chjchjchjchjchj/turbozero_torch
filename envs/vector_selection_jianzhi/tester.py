


from core.test.tester import Tester


class VectorSelectionJianzhiTester(Tester):
    def add_evaluation_metrics(self, episodes):
        if self.history is not None:
            for episode in episodes:
                moves = len(episode)
                self.history.add_evaluation_data({
                    'reward': moves,
                }, log=self.log_results)