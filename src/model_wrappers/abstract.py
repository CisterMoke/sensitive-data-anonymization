import abc
from typing import TypeVar

from src.data_models.ner import Span


class AbstractModelWrapper(abc.ABC):

    @abc.abstractmethod
    def predict(self, text: str) -> list[Span]:
        ...

ModelWrapper = TypeVar('ModelWrapper', bound=AbstractModelWrapper)