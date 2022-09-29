import numpy as np
import re
from transformers import AutoTokenizer
from functools import wraps
from pathlib import Path
import logging


def call_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger()
        logger.debug("Start call %s" % str(func.__name__))
        logger.debug(f'Arguments passed: {args}')
        logger.debug(f'{kwargs}')
        res = func(*args, **kwargs)
        logger.debug("End call %s" % str(func))
        return res
    return wrapper


def order_files(path, subset):
    files_path = Path(path)
    files = list(filter(lambda y: (y.is_file() and subset in str(y)), files_path.iterdir()))
    files = [str(_) for _ in files]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))
    return files


def split_matricies_and_lengths(adj_matricies, ntokens_array, num_of_workers):
    splitted_adj_matricies = np.array_split(adj_matricies, num_of_workers)
    splitted_ntokens = np.array_split(ntokens_array, num_of_workers)
    assert all([len(m) == len(n) for m, n in zip(splitted_adj_matricies, splitted_ntokens)]), "Split is not valid!"
    return zip(splitted_adj_matricies, splitted_ntokens)


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_tokenizer(tokenizer_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer


def get_token_length(tokenizer_dir, batch_texts, max_seq_length,
                     padding='max_length',
                     add_special_tokens=True,
                     truncation=True, **kwargs):
    tokenizer = load_tokenizer(tokenizer_dir)
    inputs = tokenizer.batch_encode_plus(batch_texts,
                                         return_tensors='pt',
                                         add_special_tokens=add_special_tokens,
                                         max_length=max_seq_length,
                                         padding=padding,
                                         truncation=truncation
                                         )
    inputs = inputs['input_ids'].numpy()
    n_tokens = []
    indexes = np.argwhere(inputs == tokenizer.pad_token_id)
    for i in range(inputs.shape[0]):
        ids = indexes[(indexes == i)[:, 0]]
        if not len(ids):
            n_tokens.append(max_seq_length)
        else:
            n_tokens.append(ids[0, 1])
    return np.array(n_tokens)


def cutoff_matrix(matrix, ntokens):
    """Return normalized submatrix of first n_tokens"""
    matrix = matrix[:ntokens, :ntokens]
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix