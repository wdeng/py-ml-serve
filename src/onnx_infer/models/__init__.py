import numpy as np


def qa_postprocssing():
    """
    Mainly code simplified and borrowed from:
    https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/question_answering.py#L95
    """
    # TBD
    raise NotImplementedError


def qa_postprocssing_simple():
    """
    a super simplified version only extracting max start and max end
    returns nothing if start > end
    """


def normalize_logits(logits, remove_first=True):
    """
    recover from logits to softmax probabilities
    remove_first: first token <CLS> removal
    """
    # Normalize logits and spans to retrieve the answer
    logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    logits = logits / logits.sum()

    return logits