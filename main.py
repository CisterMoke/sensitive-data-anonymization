import uvicorn
from fastapi import FastAPI

from config import AppConfig, ModelConfig
from src.data_models.api import Text
from src.data_models.ner import Span
from src.services.anonymizer import Anonymizer

app = FastAPI()


@app.post("/predict")
async def calculate_production_plan(text: Text) -> list[Span]:
    model_config = ModelConfig()
    anonimyzer = Anonymizer(model_config)
    return anonimyzer.wrapper.predict(text.text)

@app.post("/anonymize")
async def calculate_production_plan(text: Text) -> str:
    model_config = ModelConfig()
    anonimyzer = Anonymizer(model_config)
    return anonimyzer.anonymize(text.text)
    

if __name__ == "__main__":
    cfg = AppConfig()
    uvicorn.run('main:app', host=cfg.host, port=cfg.port, reload=cfg.debug)