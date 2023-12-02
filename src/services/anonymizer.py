from loguru import logger

from config import ModelConfig, NLPType
from src.model_wrappers import ModelWrapper, RegexWrapper, SpacyWrapper


class Anonymizer:
    _type_map: dict[NLPType, type[ModelWrapper]] = {
        NLPType.SPACY: SpacyWrapper
    }

    def __init__(self, model_config: ModelConfig) -> None:
        self.cfg = model_config
        self.wrapper: ModelWrapper = self._load_wrapper()

    def _load_wrapper(self) -> ModelWrapper:
        logger.info(f'Loading {self.cfg.nlp_type} model "{self.cfg.nlp_name}"')
        model_class = self._type_map[self.cfg.nlp_type]
        wrapper = model_class(self.cfg.nlp_name)
        wrapper.set_regex(RegexWrapper.from_file(self.cfg.pattens))
        if (map_string:= self.cfg.nlp_label_map) is not None:
            for pair in map_string.split(','):
                source, target = pair.split('=')
                wrapper.map_label(source, target)
        return wrapper
    
    def anonymize(self, text: str) -> str:
        spans = self.wrapper.predict(text)
        for span in sorted(spans, reverse=True, key=lambda x: x.end_char):
            text = f'{text[:span.start_char]}<{span.label}>{text[span.end_char:]}'
        return text
