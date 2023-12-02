from pydantic import BaseModel


class Span(BaseModel):
    start_char: int
    end_char: int
    label: str
    text: str

    def overlaps(self, other: 'Span') -> bool:
        return not (
            self.end_char < other.start_char
            or other.end_char < self.start_char
            )