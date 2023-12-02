import json

import spacy
from spacy.pipeline import EntityRuler

from src.data_models.ner import Span


class RegexWrapper:
    def __init__(self, patterns, name=None) -> None:
        self.nlp = spacy.blank('en')
        ruler: EntityRuler = self.nlp.add_pipe('entity_ruler')
        ruler.add_patterns(patterns)
        self._name = name

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        return 'regex_wrapper'

    @classmethod
    def from_file(cls, path: str) -> 'RegexWrapper':
        with open(path) as f:
            patterns = [json.loads(line) for line in f]
        return cls(patterns)
    
    def predict(self, text) -> list[Span]:
        doc = self.nlp(text)
        spans = []
        for ent in doc.ents:
            spans.append(Span(
                start_char=ent.start_char,
                end_char=ent.end_char,
                label=ent.label_,
                text=ent.text
            ))
        return spans