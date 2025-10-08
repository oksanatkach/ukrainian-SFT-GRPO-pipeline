from transformers import TrainerCallback


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=5, min_delta=0.01, monitor_metric='rewards/mean'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_metric = monitor_metric
        self.best_value = None
        self.patience_counter = 0
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.monitor_metric not in logs:
            return

        current_value = logs[self.monitor_metric]
        self.history.append(current_value)

        # We want rewards to increase
        if self.best_value is None or current_value > self.best_value + self.min_delta:
            self.best_value = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            print(f"\nEarly stopping triggered! No improvement for {self.patience} checks.")
            print(f"Best {self.monitor_metric}: {self.best_value}")
            control.should_training_stop = True

        return control


class KLDivergenceEarlyStoppingCallback(TrainerCallback):
    def __init__(self, max_kl=0.5):
        self.max_kl = max_kl

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'objective/kl' in logs:
            if logs['objective/kl'] > self.max_kl:
                print(f"\nStopping: KL divergence too high ({logs['objective/kl']:.4f})")
                control.should_training_stop = True
        return control
