from pydantic import BaseModel


class Span(BaseModel):
    start_char: int
    end_char: int
    label: str
    text: str