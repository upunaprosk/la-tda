import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

topological_names = 's_w_e_v_c_b0_b1_m_k'.split('_')
topological_thresholds = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75]
topological_feature_names = [f'{n}_t{t}' for n in topological_names for t in topological_thresholds]

barcode_feature_names = [
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
    'h1_nb']

template_feature_names = [
    'self',
    'beginning',
    'prev',
    'next',
    'comma',
    'dot',
    'sep']


def topological_get_layer_head(features, layer, head):
    df = features[layer, head, :, :, :]
    df = np.moveaxis(df, 0, -1)
    df = np.moveaxis(df, 2, 1)
    nfeat = df.shape[1]
    nthrs = df.shape[2]
    df = df.reshape((df.shape[0], nfeat * nthrs))
    return pd.DataFrame(df, columns=topological_feature_names)


def barcode_get_layer_head(features, layer, head):
    return pd.DataFrame(features[layer, head, :, :], columns=barcode_feature_names)


def template_get_layer_head(features, layer, head):
    df = features[layer, head, :, :]
    return pd.DataFrame(df.T, columns=template_feature_names)


def load_features(dataset_name, model_dir, features_dir="./features", heads="all", layers=12,
                  max_len=64, topological_thr=6,
                  topological_features="s_w_e_v_c_b0b1_m_k"):
    """
    Load topological, barcodes and template features from features_dir for dataset_name
    Returns:
        features_frame (pandas.DataFrame)
            Frame of concatenated features; column names: feature_layer_head
    """
    # dev_s_w_e_v_c_b0b1_m_k_lists_array_6.npy
    features = pd.DataFrame()
    # path_joined = dataset_name
    topological_thrs = '_'.join([topological_features, 'lists_array', str(topological_thr)])
    topological_file = '_'.join([dataset_name, topological_thrs])
    # topological_file += len_model
    barcode_file = dataset_name + "_ripser"
    template_file = dataset_name + "_template"
    all_features_dict = {
        'topological': (topological_get_layer_head, topological_feature_names, topological_file),
        'barcode': (barcode_get_layer_head, barcode_feature_names, barcode_file),
        'template': (template_get_layer_head, template_feature_names, template_file)}
    pbar = tqdm.tqdm(total=len(all_features_dict) * 12 * layers, desc="Loading {} features...".format(dataset_name),
                     position=0, leave=True)
    for feature_type, read_args in all_features_dict.items():
        function_, feature_names, path = read_args
        features_type_i_path = Path(features_dir).joinpath(*[path + '.npy'])
        features_table = np.load(features_type_i_path)
        for (layer, head) in itertools.product(range(12), range(layers)):
            X = function_(features_table, layer, head)
            columns = X.columns
            columns_names = {i: i + '_' + str(layer) + '_' + str(head) for i in columns}
            X = X.rename(columns=columns_names)
            features = pd.concat([features, X], axis=1)
            pbar.update(1)
    pbar.close()
    return features


def read_labels(set_name, data_dir="./data", file_type=".csv"):
    header = 'infer'
    names = None
    delimeter = ","
    if "tsv" in file_type:
        header = None
        names = ["Source", "label", "code", "sentence"]
        delimeter = "\t"
    df = pd.read_csv(data_dir + set_name + file_type, header=header, names=names, delimiter=delimeter)
    return df["sentence"], df["label"].values
