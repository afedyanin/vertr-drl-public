from optuna.samplers import BaseSampler, RandomSampler, TPESampler


class SamplerFactory:

    @staticmethod
    def create(
            n_startup_trials: int,
            seed: int | None = None,
            sampler_method: str = "tpe",
            ) -> BaseSampler:
        if sampler_method == "random":
            sampler: BaseSampler = RandomSampler(seed=seed)
        elif sampler_method == "tpe":
            sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed, multivariate=True)
        else:
            raise ValueError(f"Unknown sampler: {sampler_method}")
        return sampler
