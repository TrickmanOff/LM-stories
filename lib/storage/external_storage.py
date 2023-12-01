import logging
from abc import abstractmethod
from typing import List

from lib.storage.experiments_storage import ExperimentsStorage, RunStorage


logger = logging.getLogger(__name__)


def try_to_call_function(function, times: int = 3):
    exc = None
    for _ in range(times):
        try:
            function()
            return
        except Exception as e:
            exc = e
    logging.warning(f"Failed to use external storage: {exc}")


class ExternalStorage:
    def import_config(self, run_storage: RunStorage) -> None:
        try_to_call_function(lambda: self._import_config(run_storage))

    def import_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        try_to_call_function(lambda: self._import_checkpoint(run_storage, checkpoint_name))

    def export_config(self, run_storage: RunStorage) -> None:
        try_to_call_function(lambda: self._export_config(run_storage))

    def export_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        try_to_call_function(lambda: self._export_checkpoint(run_storage, checkpoint_name))

    def import_encoder(self, experiments_storage: ExperimentsStorage, encoder_name: str) -> None:
        try_to_call_function(lambda: self._import_encoder(experiments_storage, encoder_name))

    def export_encoder(self, experiments_storage: ExperimentsStorage, encoder_name: str) -> None:
        logging.info(f'Encoder {encoder_name} was successfully exported')
        try_to_call_function(lambda: self._export_encoder(experiments_storage, encoder_name))

    @abstractmethod
    def get_available_encoders(self) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def _import_config(self, run_storage: RunStorage) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _import_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _export_config(self, run_storage: RunStorage) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _export_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _import_encoder(self, experiments_storage: ExperimentsStorage, encoder_name: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _export_encoder(self, experiments_storage: ExperimentsStorage, encoder_name: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def list_content(self) -> str:
        raise NotImplementedError()
