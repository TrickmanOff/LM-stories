import os
from abc import abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import Dict, Any, Set, Optional, Tuple, List

import wandb


@dataclass
class LoggerConfig:
    ignored_periods: Optional[Set[str]] = None
    ignored_metrics: Optional[Set[str]] = None


class MLLogger:
    """An abstract class for ML loggers"""
    def __init__(self, config: Optional[LoggerConfig] = None) -> None:
        self.config = LoggerConfig(ignored_periods=set('batch'))
        self.accumulated_data: Dict[str, Tuple[int, Dict[str, Any]]] = {}   # (period: data with {period: period_index})

    def log_metrics(self, data: Dict[str, Any], period: str, period_index: Optional[int] = None, commit: bool = True) -> None:
        """
        Log values of metrics after some period

        :param data: a dict of metrics values {metric_name: value}
        :param period: the name of a period (e.g., "batch", "epoch")
        :param period_index: the index of a period, if the call is not the first for this period, it may be omitted
        :param commit: if False, data will be accumulated but not logged
        use commit=True only for the last call for the pair (period, period_index)
        """
        if self.config.ignored_periods and period in self.config.ignored_periods:
            return

        if self.config.ignored_metrics is not None:
            data = copy(data)
            for metric in copy(data):
                if metric in self.config.ignored_metrics:
                    data.pop(metric)

        data = copy(data)
        if period in self.accumulated_data:
            prev_period_index, prev_data = self.accumulated_data[period]
            if period_index is not None and prev_period_index != period_index:
                raise RuntimeError(f'Trying to log data for the {period} #{period_index} while the data for the {period} #{prev_period_index} was not logged')
            period_index = prev_period_index
            data.update(prev_data)

        assert period_index is not None, 'Period index is not specified'
        self.accumulated_data[period] = (period_index, data)

        if commit:
            self._log_metrics(data, period, period_index)
            self.accumulated_data.pop(period)

    def commit(self, period: str):
        if period in self.accumulated_data:
            self.log_metrics(data={}, period=period, commit=True)

    @abstractmethod
    def _log_metrics(self, data: Dict[str, Any], period: str, period_index: int) -> None:
        raise NotImplementedError()

    # optional for implementation
    def log_table(self, name, columns, data) -> None:
        raise NotImplementedError()


class _WandbLogger(MLLogger):
    """
    Should be used only from WandbCM
    """
    def __init__(self, config: Optional[LoggerConfig] = None):
        super().__init__(config=config)

    def _log_metrics(self, data: Dict[str, Any], period: str, period_index: int) -> None:
        # if period not in self._periods:
        wandb.define_metric(period)
        for metric in data:
            wandb.define_metric(metric, step_metric=period)
        logged_dict = copy(data)
        logged_dict[period] = period_index
        wandb.log(logged_dict)

    def log_table(self, name, columns, data) -> None:
        table = wandb.Table(columns=columns, data=data)
        wandb.log({name: table})
    # def log_generated_texts(self, texts: List[str]) -> None:
    #     data = [[text] for text in texts]
    #     text_table = wandb.Table(columns=["text"], data=data)
    #     wandb.log({'Generated texts': text_table})


def get_wandb_token(env_var_name: str = 'WANDB_TOKEN') -> str:
    is_kaggle = 'KAGGLE_URL_BASE' in os.environ
    if is_kaggle:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret(env_var_name)
    else:
        if 'WANDB_TOKEN' not in os.environ:
            raise RuntimeError('Env variable "WANDB_TOKEN" with WandB token is not defined')
        token = os.environ[env_var_name]
    return token


class WandbCM:
    """
    Wandb logger context manager

    calls wandb.login(), wandb.init() and wandb.finish()
    """
    def __init__(self, project_name: str, experiment_id: str, token: Optional[str] = None, token_env_var_name: str = 'WANDB_TOKEN',
                 config: Optional[LoggerConfig] = None) -> None:
        self.project_name = project_name
        self.experiment_id = experiment_id
        if token is None:
            token = get_wandb_token(token_env_var_name)
        self.token = token
        self.config = config

    def __enter__(self) -> _WandbLogger:
        wandb.login(key=self.token)
        wandb.init(
            project=self.project_name,
            name=self.experiment_id
        )
        return _WandbLogger()

    def __exit__(self, exc_type, exc_val, exc_tb):
        wandb.finish()

