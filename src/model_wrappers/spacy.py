import spacy

from src.data_models.ner import Span
from src.model_wrappers import AbstractModelWrapper, RegexWrapper

class SpacyWrapper(AbstractModelWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._name = kwargs.pop('_name', None)
        self.nlp = spacy.load(*args, **kwargs)

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        meta = self.nlp.meta
        return f'spacy_{meta["lang"]}_{meta["name"]}'

    def predict(self, text: str) -> list[Span]:
        doc = self.nlp(text)
        spans: list[Span] = []
        for ent in doc.ents:
            spans.append(Span(
                start_char=ent.start_char,
                end_char=ent.end_char,
                label=self.get_label(ent.label_),
                text=ent.text
                ))
        if self.regex is not None:
            spans += self.regex.predict(text)
            spans = self.remove_overlaps(spans)
        return spans
