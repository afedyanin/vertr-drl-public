from optuna.pruners import BasePruner, MedianPruner, NopPruner, SuccessiveHalvingPruner


class PrunerFactory:

    @staticmethod
    def create(
            n_startup_trials: int,
            n_evaluations: int,
            pruner_method: str = "median") -> BasePruner:
        if pruner_method == "halving":
            pruner: BasePruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif pruner_method == "median":
            pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3)
        elif pruner_method == "none":
            pruner = NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {pruner_method}")
        return pruner
