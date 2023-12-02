import abc
from typing import TypeVar

from src.data_models.ner import Span
from src.model_wrappers.regex import RegexWrapper


class AbstractModelWrapper(abc.ABC):
    def __init__(self) -> None:
        self.regex: RegexWrapper = None
        self.label_map: dict[str, str] = dict()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def predict(self, text: str) -> list[Span]:
        ...

    def set_regex(self, regex: RegexWrapper):
        self.regex: RegexWrapper = regex
        return self
    
    def map_label(self, source: str, target: str):
        self.label_map[source] = target
        return self

    def get_label(self, label) -> str:
        return self.label_map.get(label, label)

    def remove_overlaps(self, spans: list[Span]):
        if len(spans) < 2:
            return spans
        
        filtered = spans[0:1]
        for span in spans[1:]:
            prev = filtered[-1]
            if not span.overlaps(prev):
                filtered.append(span)
            elif span.label == prev.label and span.end_char > prev.end_char:
                offset = prev.end_char - span.start_char
                new_text = f'{prev.text}{span.text[offset:]}'
                filtered[-1] = Span(
                    start_char=prev.start_char,
                    end_char=span.end_char,
                    label=span.label,
                    text=new_text
                )
        return filtered

ModelWrapper = TypeVar('ModelWrapper', bound=AbstractModelWrapper)