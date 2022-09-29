from .feature_calc_utils import *
from datasets import load_dataset
import multiprocessing
from tqdm import tqdm
import json
import itertools
from collections import defaultdict
import ripserplusplus as rpp
import pathlib
import argparse


###################################
# RIPSER FEATURE CALCULATION FORMAT
###################################
# Format: "h{dim}\_{type}\_{args}"

# Dimension: 0, 1, etc.; homology dimension

# Types:

#     1. s: sum of lengths; example: "h1_s".
#     2. m: mean of lengths; example: "h1_m"
#     3. v: variance of lengths; example "h1_v"
#     4. e: entropy of persistence diagram.
#     2. n: number of barcodes with time of birth/death more/less then threshold.
#         2.1. b/d: birth or death
#         2.2. m/l: more or less than threshold
#         2.2. t: threshold value
#        example: "h0_n_d_m_t0.5", "h1_n_b_l_t0.75"
#     3. t: time of birth/death of the longest barcode (not incl. inf).
#         3.1. b/d: birth of death
#             example: "h0_t_d", "h1_t_b"

####################################

def barcode_pop_inf(barcode):
    """Delete all infinite barcodes"""
    for dim in barcode:
        if len(barcode[dim]):
            barcode[dim] = barcode[dim][barcode[dim]['death'] != np.inf]
    return barcode


def barcode_number(barcode, dim=0, bd='death', ml='m', t=0.5):
    """Calculate number of barcodes in h{dim} with time of birth/death more/less then threshold"""
    if len(barcode[dim]):
        if ml == 'm':
            return np.sum(barcode[dim][bd] >= t)
        elif ml == 'l':
            return np.sum(barcode[dim][bd] <= t)
        else:
            raise Exception("Wrong more/less type in barcode_number calculation")
    else:
        return 0.0


def barcode_time(barcode, dim=0, bd='birth'):
    """Calculate time of birth/death in h{dim} of longest barcode"""
    if len(barcode[dim]):
        max_len_idx = np.argmax(barcode[dim]['death'] - barcode[dim]['birth'])
        return barcode[dim][bd][max_len_idx]
    else:
        return 0.0


def barcode_number_of_barcodes(barcode, dim=0):
    return len(barcode[dim])


def barcode_entropy(barcode, dim=0):
    if len(barcode[dim]):
        lengths = barcode[dim]['death'] - barcode[dim]['birth']
        lengths /= np.sum(lengths)
        return -np.sum(lengths * np.log(lengths))
    else:
        return 0.0


def barcode_lengths(barcode, dim=0):
    return barcode[dim]['death'] - barcode[dim]['birth']


def barcode_sum(barcode, dim=0):
    """Calculate sum of lengths of barcodes in h{dim}"""
    if len(barcode[dim]):
        return np.sum(barcode[dim]['death'] - barcode[dim]['birth'])
    else:
        return 0.0


def barcode_mean(barcode, dim=0):
    """Calculate mean of lengths of barcodes in h{dim}"""
    if len(barcode[dim]):
        return np.mean(barcode[dim]['death'] - barcode[dim]['birth'])
    else:
        return 0.0


def barcode_std(barcode, dim=0):
    """Calculate std of lengths of barcodes in h{dim}"""
    if len(barcode[dim]):
        return np.std(barcode[dim]['death'] - barcode[dim]['birth'])
    else:
        return 0.0


def count_ripser_features(barcodes, feature_list=['h0_m']):
    """Calculate all provided ripser features"""
    # first pop all infs from barcodes
    barcodes = [barcode_pop_inf(barcode) for barcode in barcodes]
    # calculate features
    features = []
    for feature in feature_list:
        feature = feature.split('_')
        # dimension, feature type and args
        dim, ftype, fargs = int(feature[0][1:]), feature[1], feature[2:]
        if ftype == 's':
            feat = [barcode_sum(barcode, dim) for barcode in barcodes]
        elif ftype == 'm':
            feat = [barcode_mean(barcode, dim) for barcode in barcodes]
        elif ftype == 'v':
            feat = [barcode_std(barcode, dim) for barcode in barcodes]
        elif ftype == 'n':
            bd, ml, t = fargs[0], fargs[1], float(fargs[2][1:])
            if bd == 'b':
                bd = 'birth'
            elif bd == 'd':
                bd = 'death'
            feat = [barcode_number(barcode, dim, bd, ml, t) for barcode in barcodes]
        elif ftype == 't':
            bd = fargs[0]
            if bd == 'b':
                bd = 'birth'
            elif bd == 'd':
                bd = 'death'
            feat = [barcode_time(barcode, dim, bd) for barcode in barcodes]
        elif ftype == 'nb':
            feat = [barcode_number_of_barcodes(barcode, dim) for barcode in barcodes]
        elif ftype == 'e':
            feat = [barcode_entropy(barcode, dim) for barcode in barcodes]
        features.append(feat)
    return np.swapaxes(np.array(features), 0, 1)  # samples X n_features


def matrix_to_ripser(matrix, ntokens, lower_bound=0.0):
    """Convert matrix to appropriate ripser++ format"""
    matrix = cutoff_matrix(matrix, ntokens)
    matrix = (matrix > lower_bound).astype(np.int) * matrix

    matrix = 1.0 - matrix
    matrix -= np.diag(np.diag(matrix))  # 0 on diagonal
    matrix = np.minimum(matrix.T, matrix)  # symmetrical, edge emerges if at least one direction is working
    return matrix


def run_ripser_on_matrix(matrix, dim):
    barcode = rpp.run(f"--format distance --dim {dim}", data=matrix)
    return barcode


def get_barcodes(matricies, ntokens_array, dim=1, lower_bound=0.0, layer_head=(0, 0)):
    """Get barcodes from matrix"""
    barcodes = []
    layer, head = layer_head

    for i, matrix in enumerate(matricies):
        #         with open("log.txt", 'w') as fp: # logging into file
        #             fp.write(str(layer) + "_" + str(head) + "_" + str(i) + "\n")
        matrix = matrix_to_ripser(matrix, ntokens_array[i], lower_bound)
        barcode = run_ripser_on_matrix(matrix, dim)
        barcodes.append(barcode)
    return barcodes


@call_func
def get_only_barcodes(adj_matricies, ntokens_array, dim, lower_bound, verbose=False):
    """Get barcodes from adj matricies for each layer, head"""
    barcodes = {}
    layers, heads = range(adj_matricies.shape[1]), range(adj_matricies.shape[2])
    iter = itertools.product(layers, heads)
    if verbose:
        iter = tqdm(iter, 'Layer, Head', leave=False)
    for (layer, head) in iter:
        matricies = adj_matricies[:, layer, head, :, :]
        barcodes[(layer, head)] = get_barcodes(matricies, ntokens_array, dim, lower_bound, (layer, head))
    return barcodes


def format_barcodes(barcodes):
    """Reformat barcodes to json-compatible format"""
    return [{d: b[d].tolist() for d in b} for b in barcodes]


def save_barcodes(barcodes, filename):
    """Save barcodes to file"""
    formatted_barcodes = defaultdict(dict)
    for layer, head in barcodes:
        formatted_barcodes[layer][head] = format_barcodes(barcodes[(layer, head)])
    json.dump(formatted_barcodes, open(filename, 'w'))


def unite_barcodes(barcodes, barcodes_part):
    """Unite 2 barcodes"""
    for (layer, head) in barcodes_part:
        barcodes[(layer, head)].extend(barcodes_part[(layer, head)])
    return barcodes


def compute_ripser_features(model_dir, data_file, debug=False, **kwargs):
    logger = logging.getLogger()
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)
    subset = pathlib.Path(data_file).stem
    logger.debug(f"Loading barcodes from {model_dir + '/features/barcodes'} for the {subset} data subset...")
    barcode_json_files = order_files(path=model_dir + '/features/barcodes', subset=subset)
    logger.debug("Calculating features of barcode files " + "\n".join(barcode_json_files))

    features_array = []

    for filename in tqdm(barcode_json_files, desc='Calc ripser++ features'):
        barcodes = json.load(open(filename))
        logger.info(f"Barcodes loaded from: {filename}")
        features_part = []
        for layer in barcodes:
            features_layer = []
            for head in barcodes[layer]:
                ref_barcodes = reformat_barcodes(barcodes[layer][head])
                logger.debug(f"Calculating ripser++ features of saved barcodes; layer: {layer}, head: {head}")
                features = count_ripser_features(ref_barcodes, RIPSER_FEATURES)
                logger.debug(f"Calculated ripser++ features")
                features_layer.append(features)
            features_part.append(features_layer)
        features_array.append(np.asarray(features_part))
    features = np.concatenate(features_array, axis=2)
    features_dir = pathlib.Path(model_dir + '/features') / subset
    ripser_file = str(features_dir) + "_ripser" + '.npy'
    logger.debug("Saving features of the barcodes " + ripser_file)
    np.save(ripser_file, features)
    logger.info("Barcodes' features saved to: " + ripser_file)
    return


def reformat_barcodes(barcodes):
    """Return barcodes to their original format"""
    formatted_barcodes = []
    for barcode in barcodes:
        formatted_barcode = {}
        for dim in barcode:
            formatted_barcode[int(dim)] = np.asarray(
                [(b, d) for b, d in barcode[dim]], dtype=[('birth', '<f4'), ('death', '<f4')]
            )
        formatted_barcodes.append(formatted_barcode)
    return formatted_barcodes


def calculate_barcodes(model_dir, data_file, attn_iter,
                       num_of_workers=2, batch_size=10, max_seq_length=64,
                       dump_size=100, dim=1, debug=False, **kwargs):
    logger = logging.getLogger()
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)
    lower_bound = 1e-3
    data_path = pathlib.Path(data_file)
    subset = data_path.stem
    logger.debug(f"Loading {data_path.suffix[1:]} dataset from path: {str(data_path.as_posix())}...")
    dataset = load_dataset(path=data_path.suffix[1:],
                           data_files={subset: data_path.as_posix()})
    texts_column = kwargs["data_text_column"]
    logger.debug(f"Getting `n_tokens` of passed sentences. Using `{texts_column}` column of the dataset.")
    ntokens_array = get_token_length(tokenizer_dir=model_dir, batch_texts=dataset[subset]['sentence'],
                                     max_seq_length=max_seq_length, **kwargs)
    features_dir = pathlib.Path(model_dir + '/features/barcodes')
    logger.debug(f"Barcodes path: {str(features_dir)}.")
    barcodes_file = features_dir / subset
    pool = multiprocessing.get_context("spawn").Pool(num_of_workers)
    for i, filename in attn_iter:
        barcodes = defaultdict(list)
        adj_matricies = np.load(filename, allow_pickle=True)
        logger.debug(f"Matrices loaded from: {filename}")
        ntokens = ntokens_array[i * batch_size * dump_size: (i + 1) * batch_size * dump_size]
        splitted = split_matricies_and_lengths(adj_matricies, ntokens, num_of_workers)
        b_args = [(ms, nts, dim, lower_bound) for ms, nts in splitted]
        all_barcodes = pool.starmap(get_only_barcodes, b_args)
        for barcodes_part in all_barcodes:
            barcodes = unite_barcodes(barcodes, barcodes_part)
        part = filename.split('_')[-1].split('.')[0]
        save_barcodes(barcodes, str(barcodes_file) + '_' + part + '.json')
        logger.info("Saving barcodes to: " + str(barcodes_file) + '_' + part + '.json')
    return


RIPSER_FEATURES = [
    'h0_s',
    'h0_e',
    'h0_t_d',
    'h0_n_d_m_t0.75',
    'h0_n_d_m_t0.5',
    'h0_n_d_l_t0.25',
    'h1_t_b',
    'h1_n_b_m_t0.25',
    'h1_n_b_l_t0.95',
    'h1_n_b_l_t0.70',
    'h1_s',
    'h1_e',
    'h1_v',
    'h1_nb'
]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract attention weights.")
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
        help="A tokenizers `truncation` strategy: `only_first`, "
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
    calculate_barcodes(**vars(args))
    compute_ripser_features(**vars(args))
