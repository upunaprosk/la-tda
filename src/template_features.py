import multiprocessing
from tqdm.contrib.logging import logging_redirect_tqdm
import pathlib
from tqdm import tqdm
from datasets import load_dataset
import argparse
from .feature_calc_utils import *


def matrix_distance(matricies, template, broadcast=True):
    """
    Calculates the distance between the list of matricies and the template matrix.
    Args:

    -- matricies: np.array of shape (n_matricies, dim, dim)
    -- template: np.array of shape (dim, dim) if broadcast else (n_matricies, dim, dim)

    Returns:
    -- diff: np.array of shape (n_matricies, )
    """
    diff = np.linalg.norm(matricies - template, ord='fro', axis=(1, 2))
    div = np.linalg.norm(matricies, ord='fro', axis=(1, 2)) ** 2
    if broadcast:
        div += np.linalg.norm(template, ord='fro') ** 2
    else:
        div += np.linalg.norm(template, ord='fro', axis=(1, 2)) ** 2
    return diff / np.sqrt(div)


def attention_to_self(matricies):
    """
    Calculates the distance between input matricies and identity matrix,
    which representes the attention to the same token.
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.eye(n)
    return matrix_distance(matricies, template_matrix)


def attention_to_next_token(matricies):
    """
    Calculates the distance between input and E=(i, i+1) matrix,
    which representes the attention to the next token.
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.triu(np.tri(n, k=1, dtype=matricies.dtype), k=1)
    return matrix_distance(matricies, template_matrix)


def attention_to_prev_token(matricies):
    """
    Calculates the distance between input and E=(i+1, i) matrix,
    which representes the attention to the previous token.
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.triu(np.tri(n, k=-1, dtype=matricies.dtype), k=-1)
    return matrix_distance(matricies, template_matrix)


def attention_to_beginning(matricies):
    """
    Calculates the distance between input and E=(i+1, i) matrix,
    which representes the attention to [CLS] token (beginning).
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.zeros((n, n))
    template_matrix[:, 0] = 1.0
    return matrix_distance(matricies, template_matrix)


def attention_to_ids(matricies, list_of_ids, token_id):
    """
    Calculates the distance between input and ids matrix,
    which representes the attention to some particular tokens,
    which ids are in the `list_of_ids` (commas, periods, separators).
    """

    batch_size, n, m = matricies.shape
    EPS = 1e-7
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.zeros_like(matricies)
    ids = np.argwhere(list_of_ids == token_id)
    if len(ids):
        batch_ids, row_ids = zip(*ids)
        template_matrix[np.array(batch_ids), :, np.array(row_ids)] = 1.0
        template_matrix /= (np.sum(template_matrix, axis=-1, keepdims=True) + EPS)
    return matrix_distance(matricies, template_matrix, broadcast=False)


def count_template_features(matricies,
                            feature_list=['self', 'beginning', 'prev', 'next',
                                          'comma', 'dot', 'sep'],
                            sp_tokens_ids=(1010, 1012, 102),
                            ids=None, ):
    features = []
    comma_id, dot_id, sep_id = sp_tokens_ids
    logger = logging.getLogger()
    logger.info(f"Used token_ids while computing the distance-to-pattern features:"
                f" comma_id={comma_id}, dot_id={dot_id}, sep_id={sep_id}.")
    for feature in feature_list:
        if feature == 'self':
            features.append(attention_to_self(matricies))
        elif feature == 'beginning':
            features.append(attention_to_beginning(matricies))
        elif feature == 'prev':
            features.append(attention_to_prev_token(matricies))
        elif feature == 'next':
            features.append(attention_to_next_token(matricies))
        elif feature == 'comma':
            features.append(attention_to_ids(matricies, ids, comma_id))
        elif feature == 'dot':
            features.append(attention_to_ids(matricies, ids, dot_id))
        elif feature == 'sep':
            features.append(attention_to_ids(matricies, ids, sep_id))
    return np.array(features)


def get_list_of_ids(tokenizer, sentences, max_seq_length, padding='max_length',
                    add_special_tokens=True, truncation=True, **kwargs):
    inputs = tokenizer.batch_encode_plus([text_preprocessing(s) for s in sentences],
                                         add_special_tokens=add_special_tokens,
                                         max_length=max_seq_length,
                                         padding=padding,
                                         truncation=truncation
                                         )
    return np.array(inputs['input_ids'])


@call_func
def calculate_features_t(adj_matricies, template_features, sp_tokens_ids, ids=None, ):
    """Calculate template features for adj_matricies"""
    features = []
    for layer in range(adj_matricies.shape[1]):
        features.append([])
        for head in range(adj_matricies.shape[2]):
            matricies = adj_matricies[:, layer, head, :, :]
            lh_features = count_template_features(matricies=matricies,
                                                  feature_list=template_features,
                                                  ids=ids,
                                                  sp_tokens_ids=sp_tokens_ids)
            features[-1].append(lh_features)
    return np.asarray(features)


def compute_template_features(model_dir, data_file, attn_dir=None,
                              num_of_workers=2, max_seq_length=64, debug=False, **kwargs):
    logger = logging.getLogger()
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)
    data_path = pathlib.Path(data_file)

    tokenizer = load_tokenizer(model_dir)

    subset = data_path.stem
    if not attn_dir: attn_dir = model_dir + "/attentions/"
    logger.debug(f"Loading {data_path.suffix[1:]} dataset from path: {str(data_path.as_posix())}...")
    dataset = load_dataset(path=data_path.suffix[1:],
                           data_files={subset: data_path.as_posix()})
    logger.debug("Getting `n_tokens` of passed sentences. Using `sentence` column of the dataset.")
    pool = multiprocessing.get_context("spawn").Pool(num_of_workers)
    feature_list = ['self', 'beginning', 'prev', 'next', 'comma', 'dot', 'sep']
    adj_filenames = order_files(path=attn_dir, subset=subset)
    logger.debug("Using attention weights files:\n" + "\n".join(adj_filenames))
    features_array = []
    comma_id, dot_id = list(map(lambda x: tokenizer.convert_tokens_to_ids(x),
                                [",", "."]))
    sep_id = tokenizer.sep_token_id
    with logging_redirect_tqdm():
        for i, filename in enumerate(tqdm(adj_filenames, desc='Template Feature Calc')):
            adj_matricies = np.load(filename, allow_pickle=True)
            batch_size = adj_matricies.shape[0]
            sentences = np.array(dataset[subset]['sentence'][i * batch_size:(i + 1) * batch_size])
            logger.debug("Sentence samples: " + "\n".join(sentences[:2]))
            splitted_indexes = np.array_split(np.arange(batch_size), num_of_workers)
            splitted_list_of_ids = [
                get_list_of_ids(tokenizer, sentences[indx], max_seq_length=max_seq_length, **kwargs)
                for indx in splitted_indexes
            ]
            splitted_adj_matricies = [adj_matricies[indx] for indx in splitted_indexes]
            t_args = [(m, feature_list, (comma_id, dot_id, sep_id), list_of_ids) for m, list_of_ids in
                      zip(splitted_adj_matricies, splitted_list_of_ids)]
            features_array_part = pool.starmap(
                calculate_features_t, t_args
            )
            features_array.append(np.concatenate([_ for _ in features_array_part], axis=3))

    features_array = np.concatenate(features_array, axis=3)
    logger.debug("Features array concatenated.")
    features_dir = pathlib.Path(model_dir + '/features/')
    templates_file = features_dir / subset
    templates_file = str(templates_file) + "_template" + '.npy'
    logger.debug("Saving template features to: " + templates_file)
    np.save(templates_file, features_array)
    logger.debug("Template features saved to: " + templates_file)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calc template features")
    parser.add_argument("--model_dir", type=str,
                        help="A path to a directory containing vocabulary files/tokenizer name hosted on huggingface.co/<url>.")
    parser.add_argument("--data_file", type=str, help="A csv datafile; required columns: `sentence`.")
    parser.add_argument("--attn_dir", default="", type=str, help="A directory with model attention weights saved.")
    parser.add_argument("--num_of_workers", default=2, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--dump_size", default=100, type=int)
    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--padding", default="max_length", type=str,
                        help="A padding strategy: `max_length` or `longest`")
    parser.add_argument(
        '--do_not_pad',
        action='store_true',
        help='If passed sequences are not padded.', )
    parser.add_argument(
        '--no_special_tokens',
        action='store_true',
        help='If passed `add_special_tokens` tokenizer argument is set to False. '
             'Note: template features contain by default attention-to-`CLS`/`SEP` tokens.', )
    parser.add_argument(
        '--truncation', default=True, type=str,
        help="A tokenizer's `truncation` strategy: `only_first`, "
             "`only_second`, `longest_first`, `do_not_truncate`", )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode', )
    args = parser.parse_args()
    if args.do_not_pad:
        args.padding = False
    if args.no_special_tokens:
        args.add_special_tokens = False
    if not args.attn_dir:
        args.attn_dir = args.model_dir + "/attentions/"
    compute_template_features(**vars(args))