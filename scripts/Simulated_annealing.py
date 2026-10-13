class SALearningRate:
    def __init__(self, optimizer, lr0, rf=0.7, patience=5,
                 min_lr=1e-5, ema=0.9, min_delta=0.0):
        self.opt,self.lr = optimizer, lr0
        self.rf, self.patience, self.min_lr = rf, patience, min_lr
        self.ema, self.min_delta = ema, min_delta
        self.smooth, self.best, self.bad = None, -float("inf"), 0
        self._apply()

    def _apply(self):
        for g in self.opt.param_groups:
            g["lr"] = self.lr

    def step(self, metric: float):
        """
        Call once per PPO update with the rollout's mean raw reward.
        Decays the LR by rf when the smoothed metric hasn't improved
        for patience consecutive updates.
        Returns the current learning rate.
        """
        self.smooth = metric if self.smooth is None else \
            self.ema * self.smooth + (1 - self.ema) * metric
        if self.smooth > self.best + self.min_delta:
            self.best, self.bad = self.smooth, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.lr = max(self.lr * self.rf, self.min_lr)
                self.bad = 0
                self._apply()
        return self.lr