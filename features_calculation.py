from feature_src.ripser_features import *
from feature_src.template_features import *
from feature_src.topological_features import *
from utils import *
import multiprocessing
from functools import partial
import argparse
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import pathlib
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def compute_topological_features(model_dir, data_file, attn_iter,
                                 num_of_workers=2, batch_size=10, max_seq_length=64,
                                 dump_size=100, stats_name="s_w_e_v_c_b0b1_m_k",
                                 stats_cap=500,
                                 thresholds_array=[0.025, 0.05, 0.1, 0.25, 0.5, 0.75],
                                 debug=False, **kwargs):
    global logger
    data_path = pathlib.Path(data_file)
    subset = data_path.stem
    logger.debug(f"Loading {data_path.suffix[1:]} dataset from path: {str(data_path.as_posix())}...")
    texts_column = kwargs["data_text_column"]
    dataset = load_dataset(path=data_path.suffix[1:],
                           data_files={subset: data_path.as_posix()})
    logger.debug(f"Getting `n_tokens` of passed sentences. Using `{texts_column}` column of the dataset.")
    ntokens_array = get_token_length(tokenizer_dir=model_dir, batch_texts=dataset[subset][texts_column],
                                     max_seq_length=max_seq_length, **kwargs)
    layers_of_interest = list(map(int, kwargs.get("layers").split(",")))
    thrs = len(thresholds_array)
    pool = multiprocessing.get_context("spawn").Pool(num_of_workers)
    features_dir = pathlib.Path(model_dir + '/features')
    stats_file = features_dir.as_posix() + '/' + subset + "_" + stats_name \
                 + "_lists_array_" + str(thrs) + '.npy'
    logger.debug("Topological features would be saved to: " + stats_file)
    stats_tuple_lists_array = []
    with logging_redirect_tqdm():
        for i, filename in attn_iter:
            adj_matricies = np.load(filename, allow_pickle=True)
            logger.debug(f"Matrices loaded from: {filename}")
            ntokens = ntokens_array[i * batch_size * dump_size: (i + 1) * batch_size * dump_size]
            splitted = split_matricies_and_lengths(adj_matricies, ntokens, num_of_workers)
            f_args = [(m, thresholds_array, ntokens,
                       stats_name.split("_"), stats_cap, debug) for m, ntokens in splitted]
            stats_tuple_lists_array_part = pool.starmap(
                count_top_stats, f_args)
            stats_tuple_lists_array.append(
                np.concatenate([_ for _ in stats_tuple_lists_array_part], axis=3))

    stats_tuple_lists_array = np.concatenate(stats_tuple_lists_array, axis=3)
    logger.debug("Saving topological features to: " + stats_file)
    np.save(stats_file, stats_tuple_lists_array)
    logger.info("Topological features saved to: " + stats_file)
    return


def calculate_barcodes(model_dir, data_file, attn_iter,
                       num_of_workers=2, batch_size=10, max_seq_length=64,
                       dump_size=100, dim=1, **kwargs):
    global logger
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
    with logging_redirect_tqdm():
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


def compute_ripser_features(model_dir, data_file, debug=False, **kwargs):
    global logger
    subset = pathlib.Path(data_file).stem
    logger.debug(f"Loading barcodes from {model_dir + '/features/barcodes'} for the {subset} data subset...")
    barcode_json_files = order_files(path=model_dir + '/features/barcodes', subset=subset)
    logger.debug("Calculating ripser features of barcodes: " + "\n".join(barcode_json_files))

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


def compute_template_features(model_dir, data_file, attn_iter,
                              num_of_workers=2, max_seq_length=64, debug=False, **kwargs):
    global logger
    data_path = pathlib.Path(data_file)
    tokenizer = load_tokenizer(model_dir)
    subset = data_path.stem
    logger.info(f"Loading {data_path.suffix[1:]} dataset from path: {str(data_path.as_posix())}...")
    dataset = load_dataset(path=data_path.suffix[1:],
                           data_files={subset: data_path.as_posix()})
    logger.debug("Getting `n_tokens` of passed sentences. Using `sentence` column of the dataset.")
    pool = multiprocessing.get_context("spawn").Pool(num_of_workers)
    feature_list = ['self', 'beginning', 'prev', 'next', 'comma', 'dot', 'sep']
    features_array = []
    global comma_id, dot_id, sep_id
    comma_id, dot_id = list(map(lambda x: tokenizer.convert_tokens_to_ids(x),
                                [",", "."]))
    sep_id = tokenizer.sep_token_id
    texts_column = kwargs["data_text_column"]
    with logging_redirect_tqdm():
        for i, filename in attn_iter:
            adj_matricies = np.load(filename, allow_pickle=True)
            batch_size = adj_matricies.shape[0]
            sentences = np.array(dataset[subset][texts_column][i * batch_size:(i + 1) * batch_size])
            logger.debug("Sentence samples: " + "\n".join(sentences[:2]))
            splitted_indexes = np.array_split(np.arange(batch_size), num_of_workers)
            splitted_list_of_ids = [
                get_list_of_ids(tokenizer, sentences[indx], max_seq_length=max_seq_length, **kwargs)
                for indx in splitted_indexes
            ]
            splitted_adj_matricies = [adj_matricies[indx] for indx in splitted_indexes]
            t_args = [(m, feature_list, (comma_id, dot_id, sep_id), list_of_ids) for m, list_of_ids in
                      zip(splitted_adj_matricies, splitted_list_of_ids)]
            # calculate_features_t_fixed = partial(calculate_features_t, sp_tokens_ids=(comma_id, dot_id, sep_id))
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
    logger.info("Template features saved to: " + templates_file)
    return


def main(model_dir, feature_type, debug=False, **kwargs):
    global logger
    log_level = logging.DEBUG if debug else logging.INFO
    logger = set_logger(level=log_level)
    num_of_workers = kwargs["num_of_workers"]
    if num_of_workers > multiprocessing.cpu_count():
        logger.warning(
            f"Passed `{num_of_workers}` number of CPU cores when there are only `{multiprocessing.cpu_count()}` CPUs in the system.")
        logger.warning(f"Using {multiprocessing.cpu_count()} CPUs.")
        kwargs["num_of_workers"] = multiprocessing.cpu_count()
    logger.info(f"Number of workers: {num_of_workers}")
    data_path = pathlib.Path(kwargs["data_file"])
    subset = data_path.stem
    attn_dir = kwargs["attn_dir"]
    if not kwargs["attn_dir"]: attn_dir = model_dir + "/attentions/"
    adj_filenames = order_files(path=attn_dir, subset=subset)
    adj_filenames_iter = enumerate(tqdm(adj_filenames, desc=f'Calc {feature_type} features'))
    logger.debug("Attention weights files:\n" + "\n".join(adj_filenames))
    for p in (model_dir + '/features', model_dir + '/features/barcodes'):
        features_dir = pathlib.Path(p)
        features_dir.mkdir(parents=True, exist_ok=True)
    features_calc_call = {"topological": compute_topological_features,
                          "ripser": calculate_barcodes,
                          "template": compute_template_features}
    features_calc_call[feature_type](model_dir=model_dir, attn_iter=adj_filenames_iter, debug=debug, **kwargs)
    if feature_type == "ripser":
        compute_ripser_features(model_dir=model_dir, debug=debug, **kwargs)
    return


logger = logging.getLogger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract attention weights.")
    parser.add_argument("--model_dir", type=str, help="A directory with model weights saved.")
    parser.add_argument("--data_file", type=str, help="A directory with data_file.")
    parser.add_argument("--feature_type", choices=['topological', 'ripser', 'template'], type=str)
    parser.add_argument("--attn_dir", default="", type=str, help="A directory with model attention weights saved.")
    parser.add_argument("--num_of_workers", default=2, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--dump_size", default=100, type=int)
    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument('--data_text_column', type=str, default='sentence', help='A dataset column with text')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument("--layers", default=",".join([str(x) for x in range(12)]),
                        type=str, help="A string containing layers indices, separated by a comma; e.g.`10,11`.")
    parser.add_argument("--stats_name", default="s_w_e_v_c_b0b1_m_k",
                        type=str, help="""A string containing types of features, separated by an underscore:\n 
                        "s" - number of strongly connected components\n
                        "w" - number of weakly connected components\n
                        "e" - number of edges\n
                        "v" - average vertex degree\n
                        "c" - number of (directed) simple cycles\n
                        "b0b1" - Betti numbers
                        """)
    parser.add_argument("--stats_cap", default=500, type=int)
    parser.add_argument("--thresholds_array", default="0.025,0.05,0.1,0.25,0.5,0.75",
                        type=str, help="Thresholds to be used for graph construction.")
    parser.add_argument("--dim", default=1, type=int, help="A dimension to compute persistent homology up to.")
    parser.add_argument("--padding", default="max_length", type=str,
                        help="A padding strategy: `max_length` or `longest`")
    parser.add_argument("--do_not_pad", action='store_true', help='If passed sequences are not padded.', )
    parser.add_argument(
        '--no_special_tokens',
        action='store_true',
        help='If passed `add_special_tokens` tokenizer argument is set to False. '
             'Note: template features contain by default attention-to-`CLS`/`SEP` tokens.', )
    parser.add_argument(
        '--truncation', default=True, type=str,
        help="A tokenizer's `truncation` strategy: `only_first`, "
             "`only_second`, `longest_first`, `do_not_truncate`", )
    args = parser.parse_args()
    if args.do_not_pad:
        args.padding = False
    if args.no_special_tokens:
        args.add_special_tokens = False
    if args.thresholds_array:
        args.thresholds_array = [float(s.strip()) for s in args.thresholds_array.split(",")]
    if not args.attn_dir:
        args.attn_dir = args.model_dir + "/attentions/"
    main(**vars(args))