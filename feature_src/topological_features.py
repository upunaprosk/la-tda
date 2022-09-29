from .feature_calc_utils import *
from datasets import load_dataset
import networkx as nx
import argparse
import pathlib
import multiprocessing
from tqdm import tqdm


def get_filtered_mat_list(adj_matrix, thresholds_array, ntokens):
    """
    Converts adjacency matrix with real weights into list of binary matrices.
    For each threshold, those weights of adjacency matrix, which are less than
    threshold, get "filtered out" (set to 0), remained weights are set to ones.

    Args:
        adj_matrix (np.array[float, float])
        thresholds_array (iterable[float])
        ntokens (int)

    Returns:
        filtered_matricies (list[np.array[int, int]])
    """
    filtered_matrices = []
    for thr in thresholds_array:
        filtered_matrix = adj_matrix.copy()
        filtered_matrix = cutoff_matrix(filtered_matrix, ntokens)
        filtered_matrix[filtered_matrix < thr] = 0
        filtered_matrix[filtered_matrix >= thr] = 1
        filtered_matrices.append(filtered_matrix.astype(np.int8))
    return filtered_matrices


def adj_m_to_nx_list(adj_matrix, thresholds_array, ntokens, no_mat_output=False):
    """
    Converts adjacency matrix into list of unweighted digraphs, using filtering
    process from previous function.

    Args:
        adj_matrix (np.array[float, float])
        thresholds_array (iterable[float])
        ntokens (int)

    Returns:
        nx_graphs_list (list[nx.MultiDiGraph])
        filt_mat_list(list[np.array[int, int]])

    """
    #     adj_matrix = adj_matrix[:length,:length]
    filt_mat_list = get_filtered_mat_list(adj_matrix, thresholds_array, ntokens)
    nx_graphs_list = []
    for mat in filt_mat_list:
        nx_graphs_list.append(nx.from_numpy_matrix(np.array(mat),
                                                   create_using=nx.MultiDiGraph()))
    if no_mat_output:
        return nx_graphs_list, []
    else:
        return nx_graphs_list, filt_mat_list


def adj_ms_to_nx_lists(adj_matricies, \
                       thresholds_array, \
                       ntokens_array, \
                       verbose=True, \
                       no_mat_output=False):
    """
    Executes adj_m_to_nx_list for each matrix in adj_matricies array, arranges
    the results. If verbose==True, shows progress bar.

    Args:
        adj_matricies (np.array[float, float])
        thresholds_array (iterable[float])
        verbose (bool)

    Returns:
        nx_graphs_list (list[nx.MultiDiGraph])
        filt_mat_lists (list[list[np.array[int,int]]])
    """
    graph_lists = []
    filt_mat_lists = []

    iterable = range(len(adj_matricies))
    if verbose:
        iterable = tqdm(range(len(adj_matricies)),
                        desc="Calc graphs list")
    for i in iterable:
        g_list, filt_mat_list = adj_m_to_nx_list(adj_matricies[i], \
                                                 thresholds_array, \
                                                 ntokens_array[i], \
                                                 no_mat_output=no_mat_output)
        graph_lists.append(g_list)
        filt_mat_lists.append(filt_mat_lists)

    return graph_lists, filt_mat_lists


def count_stat(g_listt_j, function=nx.weakly_connected_components, cap=500):
    stat_amount = 0
    for _ in function(g_listt_j):
        stat_amount += 1
        if stat_amount >= cap:
            break
    return stat_amount


def count_weak_components(g_listt_j, cap=500):
    return count_stat(g_listt_j, function=nx.weakly_connected_components, cap=cap)


def count_strong_components(g_listt_j, cap=500):
    return count_stat(g_listt_j, function=nx.strongly_connected_components, cap=cap)


def count_simple_cycles(g_listt_j, cap=500):
    return count_stat(g_listt_j, function=nx.simple_cycles, cap=cap)


def count_b1(g_listt_j, cap=500):
    return count_stat(g_listt_j, function=nx.cycle_basis, cap=cap)


def dim_connected_components(graph_lists, strong=False, verbose=False, cap=500):
    """
    Calculates number of connected components for each graph in list
    of lists of digraphs. If strong==True, calculates strongly connected
    components, otherwise calculates weakly connected components.
    If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        strong (bool)
        verbose (bool)

    Returns:
        w_lists (list[list[int])
    """
    w_lists = []  # len == len(w_graph_lists)
    iterable = range(len(graph_lists))
    if verbose:
        iterable = tqdm(range(len(graph_lists)),
                        desc="Calc weak comp")
    for i in iterable:
        g_list = graph_lists[i]
        w_cmp = []
        for j in range(len(g_list)):
            if strong:
                w_cmp.append(count_strong_components(g_list[j], cap=cap))
            else:
                w_cmp.append(count_weak_components(g_list[j], cap=cap))
        w_lists.append(w_cmp)
    return w_lists


def dim_simple_cycles(graph_lists, verbose, cap=500):
    """
    Calculates number of simple cycles for each graph in list
    of lists of digraphs. If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        verbose (bool)

    Returns:
        c_lists (list[list[int])
    """
    c_lists = []  # len == len(pos_w_graph_lists)
    iterable = range(len(graph_lists))
    if verbose:
        iterable = tqdm(range(len(graph_lists)),
                        desc="Calc cycles")
    for i in iterable:
        g_list = graph_lists[i]
        c = []
        for j in range(len(g_list)):
            c.append(count_simple_cycles(g_list[j], cap=cap))
        c_lists.append(c)
    if verbose:
        logger = logging.getLogger()
        flat = [x for l in c_lists for x in l]
        logger.debug('dim_simple_cycles: min=%f mean=%f max=%f,  ratio of cap = %d / %d' %
                     (
                         np.min(c_lists), np.mean(c_lists), np.max(c_lists), len([x for x in flat if x == cap]),
                         len(flat)))

    return c_lists


def dim_b1(graph_lists, verbose, cap=500):
    b1_lists = []  # len == len(pos_w_graph_lists)
    iterable = range(len(graph_lists))
    if verbose:
        iterable = tqdm(range(len(graph_lists)),
                        desc="Calc b1 (undirected graphs)")
    for i in iterable:
        g_list = graph_lists[i]
        b1 = []
        for j in range(len(g_list)):
            b1.append(count_b1(nx.Graph(g_list[j].to_undirected()), cap=cap))
        b1_lists.append(b1)
    return b1_lists


def b0_b1(graph_lists, verbose):
    b0_lists = []
    b1_lists = []  # len == len(pos_w_graph_lists)
    iterable = range(len(graph_lists))
    if verbose:
        iterable = tqdm(range(len(graph_lists)),
                        desc="Calc b0, b1")
    for i in iterable:
        g_list = graph_lists[i]
        b0 = []
        b1 = []
        for j in range(len(g_list)):
            g = nx.Graph(g_list[j].to_undirected())
            w = nx.number_connected_components(g)
            e = g.number_of_edges()
            v = g.number_of_nodes()
            b0.append(w)
            b1.append(e - v + w)
        b0_lists.append(b0)
        b1_lists.append(b1)
    return b0_lists, b1_lists


def edges_f(graph_lists, verbose):
    """
    Calculates number of edges for each graph in list
    of lists of digraphs. If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        verbose (bool)

    Returns:
        e_lists (list[list[int])
    """
    e_lists = []  # len == len(pos_w_graph_lists)
    iterable = range(len(graph_lists))
    if verbose > 2:
        iterable = tqdm(range(len(graph_lists)),
                        desc="Calc edges number")
    for i in iterable:
        g_list = graph_lists[i]
        e = []
        for j in range(len(g_list)):
            e.append(g_list[j].number_of_edges())
        e_lists.append(e)
    return e_lists


def v_degree_f(graph_lists, verbose):
    """
    Calculates number of edges for each graph in list
    of lists of digraphs. If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        verbose (bool)

    Returns:
        v_lists (list[list[int])
    """
    v_lists = []  # len == len(pos_w_graph_lists)
    iterable = range(len(graph_lists))
    if verbose > 2:
        iterable = tqdm(range(len(graph_lists)),
                        desc="Calc average vertex degree")
    for i in iterable:
        g_list = graph_lists[i]
        v = []
        for j in range(len(g_list)):
            degrees = g_list[j].degree()
            degree_values = [v for k, v in degrees]
            sum_of_edges = sum(degree_values) / float(len(degree_values))
            v.append(sum_of_edges)
        v_lists.append(v)
    return v_lists


def chordality_f(graph_lists, verbose):
    """
    Checks whether the graph is chordal or not for each graph in list of lists of graphs.
    If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        verbose (bool)

    Returns:
        ch_lists (list[list[int])
    """
    ch_lists = []  # len == len(pos_w_graph_lists)
    iterable = range(len(graph_lists))
    for i in iterable:
        g_list = graph_lists[i]
        ch = []
        for j in range(len(g_list)):
            g = nx.Graph(g_list[j].to_undirected())
            # print(g)
            # print(g.edges())
            g.remove_edges_from(nx.selfloop_edges(g))
            ch_i = nx.is_chordal(g)
            ch.append(int(ch_i))
        ch_lists.append(ch)
    return ch_lists
def max_matching_f(graph_lists, verbose):
    """
    Calculates max matching size for each graph in list
    of lists of graphs. If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        verbose (bool)

    Returns:
        max_m_lists (list[list[int])
    """
    max_m_lists = []  # len == len(pos_w_graph_lists)
    iterable = range(len(graph_lists))
    if verbose > 2:
        iterable = tqdm(range(len(graph_lists)),
                        desc="Calc max matching of the graph")
    for i in iterable:
        g_list = graph_lists[i]
        max_m = []
        for j in range(len(g_list)):
            g = nx.Graph(g_list[j].to_undirected())
            m_i = nx.maximal_matching(g)
            max_m.append(len(m_i))
        max_m_lists.append(max_m)
    return max_m_lists

@call_func
def count_top_stats(adj_matricies,
                    thresholds_array,
                    ntokens_array,
                    stats_to_count={"s", "e", "c", "v", "b0b1"},
                    stats_cap=500,
                    verbose=False):
    """
    The main function for calculating topological invariants. Unites the
    functional of all functions above.
    Args:
        adj_matricies (np.array[float, float, float, float, float])
        thresholds_array (list[float])
        stats_to_count (str)
        stats_cap (int)
        verbose (bool)
    Returns:
        stats_tuple_lists_array (np.array[float, float, float, float, float])
    """
    stats_tuple_lists_array = []

    for layer_of_interest in tqdm(range(adj_matricies.shape[1])):
        stats_tuple_lists_array.append([])
        for head_of_interest in range(adj_matricies.shape[2]):
            adj_ms = adj_matricies[:, layer_of_interest, head_of_interest, :, :]
            g_lists, _ = adj_ms_to_nx_lists(adj_ms,
                                            thresholds_array=thresholds_array,
                                            ntokens_array=ntokens_array,
                                            verbose=verbose)
            feat_lists = []
            if "s" in stats_to_count:
                feat_lists.append(dim_connected_components(g_lists,
                                                           strong=True,
                                                           verbose=verbose,
                                                           cap=stats_cap))
            if "w" in stats_to_count:
                feat_lists.append(dim_connected_components(g_lists,
                                                           strong=False,
                                                           verbose=verbose,
                                                           cap=stats_cap))
            if "e" in stats_to_count:
                feat_lists.append(edges_f(g_lists, verbose=verbose))
            if "v" in stats_to_count:
                feat_lists.append(v_degree_f(g_lists, verbose=verbose))
            if "c" in stats_to_count:
                feat_lists.append(dim_simple_cycles(g_lists,
                                                    verbose=verbose,
                                                    cap=50))

            if "b0b1" in stats_to_count:
                b0_lists, b1_lists = b0_b1(g_lists, verbose=verbose)
                feat_lists.append(b0_lists)
                feat_lists.append(b1_lists)
            if "m" in stats_to_count:
                feat_lists.append(max_matching_f(g_lists,
                                                    verbose=verbose))
            if "k" in stats_to_count:
                feat_lists.append(chordality_f(g_lists,
                                                    verbose=verbose))
            stats_tuple_lists_array[-1].append(tuple(feat_lists))

    stats_tuple_lists_array = np.asarray(stats_tuple_lists_array,
                                         dtype=np.float16)
    return stats_tuple_lists_array


def function_for_v(list_of_v_degrees_of_graph):
    return sum(map(lambda x: np.sqrt(x * x), list_of_v_degrees_of_graph))


def compute_topological_features(model_dir, data_file, attn_dir=None,
                                 num_of_workers=2, batch_size=10, max_seq_length=64,
                                 dump_size=100, stats_name="s_w_e_v_c_b0b1",
                                 stats_cap=500,
                                 thresholds_array=[0.025, 0.05, 0.1, 0.25, 0.5, 0.75],
                                 debug=False, **kwargs):
    logger = logging.getLogger()
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)
    data_path = pathlib.Path(data_file)
    subset = data_path.stem
    if not attn_dir: attn_dir = model_dir + "/attentions/"
    logger.debug(f"Loading {data_path.suffix[1:]} dataset from path: {str(data_path.as_posix())}...")
    dataset = load_dataset(path=data_path.suffix[1:],
                           data_files={subset: data_path.as_posix()})
    logger.debug("Getting `n_tokens` of passed sentences. Using `sentence` column of the dataset.")
    ntokens_array = get_token_length(tokenizer_dir=model_dir, batch_texts=dataset[subset]['sentence'],
                                     max_seq_length=max_seq_length, **kwargs)
    adj_filenames = order_files(path=attn_dir, subset=subset)
    logger.debug("Using attention weights files:\n" + "\n".join(adj_filenames))
    layers_of_interest = list(map(int, kwargs.get("layers").split(",")))
    thrs = len(thresholds_array)
    pool = multiprocessing.get_context("spawn").Pool(num_of_workers)
    logger.debug(f"Number of workers: {num_of_workers}")
    features_dir = pathlib.Path(model_dir + '/features')
    stats_file = features_dir.as_posix() + '/' + subset + "_all_heads_" + str(
        len(layers_of_interest)) + "_layers_" + stats_name \
                 + "_lists_array_" + str(thrs) + "_thrs_MAX_LEN_" + str(max_seq_length) + \
                 "_" + model_dir.split("/")[-1] + '.npy'
    logger.debug("Topological features would be saved to: " + stats_file)
    stats_tuple_lists_array = []
    for i, filename in enumerate(tqdm(adj_filenames, desc='Calculating topological features')):
        adj_matricies = np.load(filename, allow_pickle=True)
        logger.debug(f"Matricies loaded from: {filename}")
        ntokens = ntokens_array[i * batch_size * dump_size: (i + 1) * batch_size * dump_size]
        splitted = split_matricies_and_lengths(adj_matricies, ntokens, num_of_workers)
        f_args = [(m, thresholds_array, ntokens, stats_name.split("_"), stats_cap, debug) for m, ntokens in splitted]
        stats_tuple_lists_array_part = pool.starmap(
            count_top_stats, f_args)
        stats_tuple_lists_array.append(
            np.concatenate([_ for _ in stats_tuple_lists_array_part], axis=3))

    stats_tuple_lists_array = np.concatenate(stats_tuple_lists_array, axis=3)
    logger.debug("Saving topological features to: " + stats_file)
    np.save(stats_file, stats_tuple_lists_array)
    logger.info("Topological features saved to: " + stats_file)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract attention weights.")
    parser.add_argument("--model_dir", type=str, help="A directory with model weights saved.")
    parser.add_argument("--data_file", type=str, help="A csv datafile; required columns: `sentence` or `text`")
    parser.add_argument("--attn_dir", default="", type=str, help="A directory with model attention weights saved.")
    parser.add_argument("--num_of_workers", default=2, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--dump_size", default=100, type=int)
    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument("--layers", default=",".join([str(x) for x in range(12)]),
                        type=str, help="A string containing layers indices, separated by a comma; e.g.`10,11`.")
    parser.add_argument("--stats_name", default="s_w_e_v_c_b0b1",
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
    args = parser.parse_args()
    if args.thresholds_array:
        args.thresholds_array = [float(s.strip()) for s in args.thresholds_array.split(",")]

    if not args.attn_dir:
        args.attn_dir = args.model_dir + "/attentions/"
    compute_topological_features(**vars(args))
