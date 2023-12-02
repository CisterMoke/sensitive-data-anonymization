from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    host: str = Field(default='0.0.0.0')
    port: int = Field(default=8888)
    debug: bool = Field(default=False)

    model_config = SettingsConfigDict(env_prefix='app_')


class NLPType(str, Enum):
    SPACY = 'spacy'
    NLTK = 'nltk'


class ModelConfig(BaseSettings):
    nlp_type: NLPType = Field(default=NLPType.SPACY)
    nlp_name: str = Field(default='en_core_web_sm')
    pattens: str = Field(default='data/patterns.jsonl', alias='nlp_patterns')
    nlp_label_map: str|None = Field(default='GPE=LOCATION,DATE=DATE_TIME')

    model_config = SettingsConfigDict(
        use_enum_values=True, env_prefix='nlp_'
        )