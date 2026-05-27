import numpy as np

class EarlyStopping:
    def __init__(
        self,
        mode="max",        # 指标通常是越大越好
        patience=8,
        delta=0.0,
        verbose=True
    ):
        assert mode in ["min", "max"]
        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_result = {}
        self.best_model_checkpoint = None

        if self.mode == "min":
            self.is_improvement = lambda current, best: current < (best - self.delta)
            self.best_score = float("inf")
        else:
            self.is_improvement = lambda current, best: current > (best + self.delta)
            self.best_score = float("-inf")

    def step(self, current_score, current_metric, model_params):
        if self.best_score is None:
            self.best_score = current_score
            self.best_result = current_metric
            self.best_model_checkpoint = model_params
            return False

        if self.is_improvement(current_score, self.best_score):
            self.best_score = current_score
            self.best_result = current_metric
            self.best_model_checkpoint = model_params
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save(self, path):
        np.save(path, self.best_model_checkpoint)