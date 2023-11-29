import spacy

from src.data_models.ner import Span
from src.model_wrappers import AbstractModelWrapper

class SpacyWrapper(AbstractModelWrapper):
    def __init__(self, *args, **kwargs) -> None:
        self.nlp = spacy.load(*args, **kwargs)

    def predict(self, text: str) -> list[Span]:
        doc = self.nlp(text)
        spans: list[Span] = []
        for ent in doc.ents:
            spans.append(Span(
                start_char=ent.start_char,
                end_char=ent.end_char,
                label=ent.label_,
                text=ent.text
                ))
        return spans
