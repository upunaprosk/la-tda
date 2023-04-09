import argparse
import pathlib
from math import ceil

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.feature_calc_utils import text_preprocessing
from utils import *


def grab_attention_weights(model, tokenizer, sentences, max_seq_length, device='cuda:0',
                           padding='max_length',
                           add_special_tokens=True,
                           truncation=True, **kwargs):
    inputs = tokenizer.batch_encode_plus([text_preprocessing(s) for s in sentences],
                                         return_tensors='pt',
                                         add_special_tokens=add_special_tokens,
                                         max_length=max_seq_length,
                                         padding=padding,
                                         truncation=truncation
                                         )
    input_ids = inputs['input_ids'].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    attention = model(input_ids, attention_mask, token_type_ids)['attentions']
    # layer X sample X head X n_token X n_token
    attention = np.asarray([layer.cpu().detach().numpy() for layer in attention], dtype=np.float16)

    return attention


def grab_attention_weights_inference(model_dir, data_file, batch_size=10,
                                     max_seq_length=64,
                                     device='cuda:0', dump_size=100,
                                     debug=False, **kwargs):
    log_level = logging.DEBUG if debug else logging.INFO
    logger = set_logger(level=log_level)
    data_path = pathlib.Path(data_file)
    subset = data_path.stem
    logger.info(f"Loading {data_path.suffix[1:]} dataset from path: {str(data_path.as_posix())}...")
    dataset = load_dataset(path=data_path.suffix[1:],
                           data_files={subset: data_path.as_posix()})
    batch_size = batch_size
    max_seq_length = max_seq_length
    device = torch.device(device)
    logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
    logger.warning("Using %s" % device)
    logger.info("CUDA version : %s" % torch.version.cuda)
    logger.info("PyTorch version : %s" % torch.__version__)
    logger.info(f"Loading model from {model_dir}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, output_attentions=True)
    logger.info(f"Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = model.to(device)
    number_of_batches = ceil(len(dataset[subset]['sentence']) / batch_size)
    batched_sentences = np.array_split(dataset[subset]['sentence'], number_of_batches)
    number_of_files = ceil(number_of_batches / dump_size)
    adj_matricies = []
    assert (number_of_batches == len(batched_sentences)
            ), "Batch Split Error"
    attn_dir = pathlib.Path(model_dir + "/attentions")
    attn_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(number_of_batches),
                  desc="Weights Extraction"):
        attention_w = grab_attention_weights(model, tokenizer, batched_sentences[i],
                                             max_seq_length, device, **kwargs)
        adj_matricies.append(attention_w)
        if (i + 1) % dump_size == 0:
            adj_matricies = np.concatenate(adj_matricies, axis=1)
            adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1)
            file_name_i = subset + "_part" + str(ceil(i / dump_size)) + "of" + str(number_of_files) + '.npy'
            filename = str(attn_dir / file_name_i)
            logger.info(f"Saving weights to: {filename}")
            np.save(filename, adj_matricies)
            adj_matricies = []

    if len(adj_matricies):
        file_name_i = subset + "_part" + str(ceil(i / dump_size)) + "of" + str(number_of_files) + '.npy'
        filename = str(attn_dir / file_name_i)
        adj_matricies = np.concatenate(adj_matricies, axis=1)
        adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1)
        logger.info(f"Saving weights to: {filename}")
        np.save(filename, adj_matricies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract attention weights.")
    parser.add_argument("--model_dir", type=str, help="A directory with model weights saved.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether to use CUDA even when it is available or not.")
    parser.add_argument("--data_file", type=str, help="A csv datafile; required columns: `sentence` or `text`")
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--dump_size", default=100, type=int)
    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument('--debug', action='store_true', help='Debug mode')
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
    if not args.no_cuda:
        if not torch.cuda.is_available():
            raise ValueError("CUDA not available.")
    args.device = "cpu" if args.no_cuda else "cuda"
    grab_attention_weights_inference(**vars(args))
