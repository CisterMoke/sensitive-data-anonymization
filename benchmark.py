import os.path as pth
import re

import numpy as np
import pandas as pd
import spacy
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data_models.ner import Span
from src.model_wrappers import RegexWrapper, SpacyWrapper


def get_spans(data: pd.DataFrame) -> list[list[Span]]:
    all_spans = []
    label_reg = r'<[A-Z_]+>'
    for _, row in data.iterrows():
        row_spans = []
        raw, labeled = row
        raw: str
        labeled: str

        labels, split_starts, split_ends = [], [], []
        for m in re.finditer(label_reg, labeled):
            labels.append(m[0][1:-1])
            split_starts.append(m.end())
            split_ends.append(m.start())

        split_starts.insert(0, 0)
        split_ends.append(len(labeled))
        chunks = [labeled[start:end] 
                  for start, end in zip(split_starts, split_ends)]
        
        offsets = []
        find_start = 0
        for i, chunk in enumerate(chunks):
            if i == 0:
                start = 0
                offsets.append(0)
            elif chunk == "":
                start = find_start
                offsets.append(split_ends[i-1] + offsets[i-1] + 1)
            else:
                start = raw.find(chunk, find_start)
                offsets.append(start - split_starts[i])
            find_start = start + len(chunk) + 1

        for i, label in enumerate(labels):
            start_char = split_ends[i] + offsets[i]
            end_char = split_starts[i+1] + offsets[i+1]
            row_spans.append(Span(
                start_char=start_char,
                end_char=end_char,
                label=label,
                text=raw[start_char:end_char]
            ))
        all_spans.append(row_spans)
    return all_spans


def token_boundaries(text: str) -> list[tuple[int, int]]:
    nlp = spacy.blank('en')
    doc = nlp(text)
    return [(tok.idx, tok.idx + len(tok)) for tok in doc]


def overlaps(s1: tuple[int, int], s2: tuple[int, int]) -> bool:
    return not (s1[1] < s2[0] or s2[1] < s1[0])


def spans_to_labels(spans: list[Span], text: str) -> list[str]:
    tokens = token_boundaries(text)
    len_spans = len(spans)
    if len_spans == 0:
        return ['O' for t in tokens]
    
    labels: list[str] = []
    idx = 0
    curr_span = (spans[idx].start_char, spans[idx].end_char)
    for t in tokens:
        if idx >= len_spans:
            labels.append('O')
            continue
        if overlaps(t, curr_span):
            labels.append(spans[idx].label)
            while overlaps(t, curr_span) and idx < len_spans - 1:
                idx += 1
                curr_span = (spans[idx].start_char, spans[idx].end_char)
        else:
            labels.append('O')
    return labels


if __name__ == '__main__':
    logger.info('Loading bench data...')
    data_folder = pth.join(pth.dirname(__file__), 'data') 
    bench_path = pth.join(data_folder, 'benchmark.csv')
    bench_data = pd.read_csv(bench_path, header=0, sep=';')
    bench_spans = get_spans(bench_data)
    bench_labels = [
        spans_to_labels(spans, text)
        for spans, text in zip(bench_spans, bench_data['text'])
        ]

    reg_wrap = RegexWrapper.from_file(pth.join(data_folder, 'patterns.jsonl'))
    models: list[SpacyWrapper] = [
        SpacyWrapper('en_core_web_sm'),
        SpacyWrapper('en_core_web_lg'),
        SpacyWrapper(
            'en_core_web_lg', _name='en_lg_regex'
            ).set_regex(reg_wrap).map_label('GPE', 'LOCATION').map_label('DATE', 'DATE_TIME'),   
    ]

    logger.info('Calculating model scores...')
    model_scores = dict()
    for model in models:
        mdl_spans = [model.predict(text) for text in bench_data['text']]
        mdl_labels = [
            spans_to_labels(spans, text)
            for spans, text in zip(mdl_spans, bench_data['text'])
        ]
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        for x, y in zip(bench_labels, mdl_labels):
            pos_labels = list(set(x).difference({'O'}))
            
            scores['accuracy'].append(accuracy_score(x, y))
            scores['precision'].append(precision_score(
                x, y, average='micro', labels=pos_labels,
                zero_division=0.0
                ))
            scores['recall'].append(recall_score(
                x, y, average='micro', labels=pos_labels,
                zero_division=0.0
                ))
            scores['f1_score'].append(f1_score(
                x, y, average='micro', labels=pos_labels,
                zero_division=0.0
                ))
        
        model_scores[model.name] = {k: np.mean(v) for k, v in scores.items()}

    print(pd.DataFrame(model_scores))