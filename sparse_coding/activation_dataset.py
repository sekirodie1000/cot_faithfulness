import json
import math
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, Literal

import pandas as pd
import torch
import torch.nn as nn

from baukit import Trace
from datasets import Dataset, DatasetDict, load_dataset, DownloadConfig
from einops import rearrange
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_official_model_name, convert_hf_model_config
from transformers import PreTrainedTokenizerBase
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *

T = TypeVar("T", bound=Union[Dataset, DatasetDict])

MODEL_BATCH_SIZE = 4
CHUNK_SIZE_GB = 2.0
MAX_SENTENCE_LEN = 256

FEW_SHOT_COT = """\
Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: How many clips did Natalia sell in May? ** Natalia sold 48/2 = <<48/2=24>>24 clips in May.
How many clips did Natalia sell altogether in April and May? ** Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: How much does Weng earn per minute? ** Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
How much did Weng earn? ** Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
A: How much money does Betty have in the beginning? ** In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
How much money did Betty's grandparents give her? ** Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
How much more money does Betty need to buy the wallet? ** This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
#### 5

"""



def check_use_baukit(model_name):
    if model_name in ["nanoGPT"]:
        return True
    elif check_transformerlens_model(model_name):
        return False
    else:
        raise NotImplementedError(f"Unknown if model {model_name} uses baukit")


def get_activation_size(model_name: str, layer_loc: str):
    # assert check_transformerlens_model(model_name) or model_name == "nanoGPT", f"Model {model_name} not supported"
    if not (check_transformerlens_model(model_name) or model_name == "nanoGPT"):
        print(f"[INFO] Assuming HuggingFace model: {model_name}")

    assert layer_loc in [
        "residual",
        "mlp",
        "attn",
        "attn_concat",
        "mlpout",
    ], f"Layer location {layer_loc} not supported"
    model_cfg = convert_hf_model_config(model_name)
    if layer_loc == "residual":
        return model_cfg["d_model"]
    elif layer_loc == "mlp":
        return model_cfg["d_mlp"]
    elif layer_loc == "attn":
        return model_cfg["d_head"] * model_cfg["n_heads"]
    elif layer_loc == "mlpout":
        return model_cfg["d_model"]
    elif layer_loc == "attn_concat":
        return model_cfg["d_head"] * model_cfg["n_heads"]


# def check_transformerlens_model(model_name: str):
#     try:
#         get_official_model_name(model_name)
#         return True
#     except ValueError:
#         return False

def check_transformerlens_model(model_name: str):
    try:
        get_official_model_name(model_name)
        return True
    except ValueError:
        return False
    except Exception:
        return False



def make_tensor_name(layer: int, layer_loc: str, model_name: str) -> str:
    """Make the tensor name for a given layer and model."""
    assert layer_loc in [
        "residual",
        "mlp",
        "attn",
        "attn_concat",
        "mlpout",
    ], f"Layer location {layer_loc} not supported"

    if "llama" in model_name.lower():
        if layer_loc == "residual":
            return f"model.layers.{layer}.input_layernorm"
        elif layer_loc == "mlp":
            return f"model.layers.{layer}.mlp.gate_proj"
        elif layer_loc == "attn":
            return f"model.layers.{layer}.self_attn.q_proj"
        elif layer_loc == "mlpout":
            return f"model.layers.{layer}.post_attention_layernorm"
    else:
        if layer_loc == "residual":
            if check_transformerlens_model(model_name):
                tensor_name = f"blocks.{layer}.hook_resid_post"
            else:
                raise NotImplementedError(f"Model {model_name} not supported for residual stream")
        elif layer_loc == "attn_concat":
            if check_transformerlens_model(model_name):
                tensor_name = f"blocks.{layer}.attn.hook_z"
            else:
                raise NotImplementedError(f"Model {model_name} not supported for attention output")
        elif layer_loc == "mlp":
            if check_transformerlens_model(model_name):
                tensor_name = f"blocks.{layer}.mlp.hook_post"
            elif model_name == "nanoGPT":
                tensor_name = f"transformer.h.{layer}.mlp.c_fc"
            else:
                raise NotImplementedError(f"Model {model_name} not supported for MLP")
        elif layer_loc == "attn":
            if check_transformerlens_model(model_name):
                tensor_name = f"blocks.{layer}.hook_resid_post"
            else:
                raise NotImplementedError(f"Model {model_name} not supported for attention stream")
        elif layer_loc == "mlpout":
            if check_transformerlens_model(model_name):
                tensor_name = f"blocks.{layer}.hook_mlp_out"
            else:
                raise NotImplementedError(f"Model {model_name} not supported for MLP")

    return tensor_name


def read_from_pile(address: str, max_lines: int = 100_000, start_line: int = 0):
    """Reads a file from the Pile dataset. Returns a generator."""

    with open(address, "r") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if i >= max_lines + start_line:
                break
            yield json.loads(line)


def make_sentence_dataset(dataset_name: str, max_lines: int = 20_000, start_line: int = 0):
    """Returns a dataset from the Huggingface Datasets library."""
    if dataset_name == "EleutherAI/pile":
        if not os.path.exists("pile0"):
            print("Downloading shard 0 of the Pile dataset (requires 50GB of disk space).")
            if not os.path.exists("pile0.zst"):
                os.system("curl https://the-eye.eu/public/AI/pile/train/00.jsonl.zst > pile0.zst")
                os.system("unzstd pile0.zst")
        dataset = Dataset.from_list(list(read_from_pile("pile0", max_lines=max_lines, start_line=start_line)))
    elif dataset_name == "openai/gsm8k":
        print("Loading openai/gsm8k dataset...")
        dataset = load_dataset(
            dataset_name,
            split="train",
            verification_mode="no_checks",
            download_mode="reuse_dataset_if_exists"
        )

        # dataset = dataset.map(lambda x: {"text": x["question"]})
        dataset = dataset.map(lambda x: {"text": x["question"]}, remove_columns=[])
    else:
        #download_config = DownloadConfig(read_timeout=300, retries=5)
        dataset = load_dataset(dataset_name, split="train")#, split=f"train[{start_line}:{start_line + max_lines}]")
    return dataset

def make_sentence_dataset_with_cot(dataset_name: str, max_lines: int = 20_000, start_line: int = 0):
    print("DATASET + CoT")
    dataset = make_sentence_dataset(dataset_name, max_lines=max_lines, start_line=start_line)
    def prepend_cot_prompt(example):
        return {"text": FEW_SHOT_COT + f"Q: {example['text']}\nA:"}
    return dataset.map(prepend_cot_prompt)

def make_sentence_dataset_with_Nocot(dataset_name: str, max_lines: int = 20_000, start_line: int = 0):
    print("DATASET + NoCoT")
    dataset = make_sentence_dataset(dataset_name, max_lines=max_lines, start_line=start_line)
    def prepend_Nocot_prompt(example):
        return {"text": f"Q: {example['text']}\nA:"}
    return dataset.map(prepend_Nocot_prompt)




# Nora's Code from https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/data.py
def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = min(mp.cpu_count() // 2, 8),
    text_key: str = "text",
    max_length: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
) -> Tuple[T, float]:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_length` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_length: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.

    Returns:
        * The chunked and tokenized dataset.
        * The ratio of nats to bits per byte see https://arxiv.org/pdf/2101.00027.pdf,
            section 3.1.
    """

    def _tokenize_fn(x: Dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_length)  # tokenizer max length is 1024 for gpt2
        sep = tokenizer.eos_token or "<|endoftext|>"
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output["input_ids"][0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size] for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        total_tokens = sum(len(ids) for ids in output["input_ids"])
        total_bytes = len(joined_text.encode("utf-8"))

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single batch complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        # We need to output this in order to compute the number of bits per byte
        div, rem = divmod(total_tokens, output_batch_size)
        output["length"] = [div] * output_batch_size
        output["length"][-1] += rem

        div, rem = divmod(total_bytes, output_batch_size)
        output["bytes"] = [div] * output_batch_size
        output["bytes"][-1] += rem

        return output

    data = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    total_bytes: float = sum(data["bytes"])
    total_tokens: float = sum(data["length"])
    return data.with_format(format, columns=["input_ids"]), (total_tokens / total_bytes) / math.log(2)


def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> List[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names


# End Nora's Code from https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/data.py


def make_activation_dataset(
    sentence_dataset: DataLoader,
    model: HookedTransformer,
    tensor_name: str,
    activation_width: int,
    dataset_folder: str,
    baukit: bool = False,
    chunk_size_gb: float = 2,
    device: torch.device = torch.device("cuda:0"),
    layer: int = 2,
    n_chunks: int = 1,
    max_length: int = 256,
    model_batch_size: int = 4,
    center_dataset: bool = False
) -> pd.DataFrame:
    print(f"Running model and saving activations to {dataset_folder}")
    with torch.no_grad():
        chunk_size = chunk_size_gb * (2**30)  # 2GB
        activation_size = (
            activation_width * 2 * model_batch_size * max_length
        )  # 3072 mlp activations, 2 bytes per half, 1024 context window
        actives_per_chunk = chunk_size // activation_size
        dataset = []
        n_saved_chunks = 0
        for batch_idx, batch in tqdm(enumerate(sentence_dataset)):
            batch = batch["input_ids"].to(device)
            if baukit:
                # Don't have nanoGPT models integrated with transformer_lens so using baukit for activations
                with Trace(model, tensor_name) as ret:
                    _ = model(batch)
                    mlp_activation_data = ret.output
                    mlp_activation_data = rearrange(mlp_activation_data, "b s n -> (b s) n").to(torch.float16).to(device)
                    mlp_activation_data = nn.functional.gelu(mlp_activation_data)
            else:
                _, cache = model.run_with_cache(batch, stop_at_layer=layer + 1)
                mlp_activation_data = (
                    cache[tensor_name].to(device).to(torch.float16)
                )  # NOTE: could do all layers at once, but currently just doing 1 layer
                mlp_activation_data = rearrange(mlp_activation_data, "b s n -> (b s) n")

            dataset.append(mlp_activation_data)
            if len(dataset) >= actives_per_chunk:
                if center_dataset:
                    if n_saved_chunks == 0:
                        chunk_mean = torch.mean(torch.cat(dataset), dim=0)
                    dataset = [x - chunk_mean for x in dataset]
                    
                # Need to save, restart the list
                save_activation_chunk(dataset, n_saved_chunks, dataset_folder)
                n_saved_chunks += 1
                print(f"Saved chunk {n_saved_chunks} of activations, total size:  {batch_idx * activation_size} ")
                dataset = []
                if n_saved_chunks == n_chunks:
                    break

        if n_saved_chunks < n_chunks:
            save_activation_chunk(dataset, n_saved_chunks, dataset_folder)
            print(f"Saved undersized chunk {n_saved_chunks} of activations, total size:  {batch_idx * activation_size} ")


# def make_activation_dataset_tl(
#     sentence_dataset: DataLoader,
#     model: HookedTransformer,
#     activation_width: int,
#     dataset_folders: List[str],
#     layers: List[int] = [2],
#     tensor_loc: str = "residual",
#     chunk_size_gb: float = 2,
#     device: torch.device = torch.device("cuda:0"),
#     n_chunks: int = 1,
#     max_length: int = 256,
#     model_batch_size: int = 4,
#     skip_chunks: int = 0,
#     center_dataset: bool = False
# ):
    
#     with torch.no_grad():
#         chunk_size = chunk_size_gb * (2**30)  # 2GB
#         activation_size = (
#             activation_width * 2 * model_batch_size * max_length
#         )  # 3072 mlp activations, 2 bytes per half, 1024 context window
#         max_batches_per_chunk = int(chunk_size // activation_size)
#         if center_dataset:
#             chunk_means = {}

#         batches_to_skip = skip_chunks * max_batches_per_chunk

#         dataset_iterator = iter(sentence_dataset)

#         n_activations = 0

#         for _ in range(batches_to_skip):
#             dataset_iterator.__next__()

#         for chunk_idx in range(n_chunks):
#             datasets: Dict[int, List] = {layer: [] for layer in layers}
#             for batch_idx, batch in tqdm(enumerate(dataset_iterator)):
#                 batch = batch["input_ids"].to(device)
#                 _, cache = model.run_with_cache(batch, stop_at_layer=max(layers) + 1)
#                 for layer in layers:
#                     tensor_name = make_tensor_name(layer, tensor_loc, model.cfg.model_name)
#                     activation_data = cache[tensor_name].to(torch.float16)
#                     if tensor_loc == "attn_concat":
#                         activation_data = rearrange(activation_data, "b s n d -> (b s) (n d)")
#                     else:
#                         activation_data = rearrange(activation_data, "b s n -> (b s) n")
#                     if layer == layers[0]:
#                         n_activations += activation_data.shape[0]
#                     datasets[layer].append(activation_data)

#                 if batch_idx >= max_batches_per_chunk:
#                     break

#             for layer, folder in zip(layers, dataset_folders):
#                 dataset = datasets[layer]
#                 if center_dataset:
#                     if chunk_idx == 0:
#                         chunk_means[layer] = torch.mean(torch.cat(dataset), dim=0)
#                     dataset = [x - chunk_means[layer]  for x in dataset]
#                 save_activation_chunk(dataset, chunk_idx, folder)

#             if len(datasets[layer]) < max_batches_per_chunk:
#                 print(f"Saved undersized chunk {chunk_idx} of activations, total size: {batch_idx * activation_size}")
#                 break
#             else:
#                 print(f"Saved chunk {chunk_idx} of activations, total size: {(chunk_idx + 1) * batch_idx * activation_size}")
    
#     #return ((chunk_means, chunk_stds) if center_dataset else None, n_activations)
#     return n_activations

from pathlib import Path

def make_activation_dataset_tl(
    sentence_dataset: DataLoader,
    model: HookedTransformer,
    activation_width: int,
    dataset_folders: List[str],
    layers: List[int] = [2],
    tensor_loc: str = "residual",
    chunk_size_gb: float = 2,
    device: torch.device = torch.device("cuda:0"),
    n_chunks: int = 1,
    max_length: int = 256,
    model_batch_size: int = 4,
    skip_chunks: int = 0,
    center_dataset: bool = False
):
    with torch.no_grad():
        chunk_size = chunk_size_gb * (2**30)
        activation_size = (
            activation_width * 2 * model_batch_size * max_length
        )
        max_batches_per_chunk = int(chunk_size // activation_size)
        if center_dataset:
            chunk_means = {}

        batches_to_skip = skip_chunks * max_batches_per_chunk
        dataset_iterator = iter(sentence_dataset)
        n_activations = 0

        for _ in range(batches_to_skip):
            next(dataset_iterator)

        for chunk_idx in range(n_chunks):
            datasets: Dict[int, List] = {layer: [] for layer in layers}
            input_ids_storage = []

            for batch_idx, batch in tqdm(enumerate(dataset_iterator)):
                input_ids = batch["input_ids"].to(device)
                input_ids_storage.append(input_ids.cpu())
                _, cache = model.run_with_cache(input_ids, stop_at_layer=max(layers) + 1)
                for layer in layers:
                    tensor_name = make_tensor_name(layer, tensor_loc, model.cfg.model_name)
                    activation_data = cache[tensor_name].to(torch.float16)
                    if tensor_loc == "attn_concat":
                        activation_data = rearrange(activation_data, "b s n d -> (b s) (n d)")
                    else:
                        activation_data = rearrange(activation_data, "b s n -> (b s) n")
                    if layer == layers[0]:
                        n_activations += activation_data.shape[0]
                    datasets[layer].append(activation_data)

                if batch_idx >= max_batches_per_chunk:
                    break

            for layer, folder in zip(layers, dataset_folders):
                dataset = datasets[layer]
                if center_dataset:
                    if chunk_idx == 0:
                        chunk_means[layer] = torch.mean(torch.cat(dataset), dim=0)
                    dataset = [x - chunk_means[layer] for x in dataset]
                save_activation_chunk(dataset, chunk_idx, folder)

                input_ids_tensor = torch.cat(input_ids_storage, dim=0)
                input_ids_path = Path(folder) / f"input_ids_layer{layer}_chunk{chunk_idx}.pt"
                torch.save(input_ids_tensor, input_ids_path)

            if len(datasets[layer]) < max_batches_per_chunk:
                print(f"Saved undersized chunk {chunk_idx} of activations, total size: {batch_idx * activation_size}")
                break
            else:
                print(f"Saved chunk {chunk_idx} of activations, total size: {(chunk_idx + 1) * batch_idx * activation_size}")

    return n_activations


def make_activation_dataset_hf(
    sentence_dataset: Dataset,
    model: AutoModelForCausalLM,
    tensor_names: List[str],
    chunk_size: int,
    n_chunks: int,
    output_folder: str = "activation_data",
    skip_chunks: int = 0,
    device: Optional[torch.device] = torch.device("cuda:0"),
    max_length: int = 2048,
    model_batch_size: int = 4,
    precision: Literal["float16", "float32"] = "float16",
    shuffle_seed: Optional[int] = None,
):
    with torch.no_grad():
        model.eval()

        dtype = None
        if precision == "float16":
            dtype = torch.float16
        elif precision == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f"Invalid precision '{precision}'")

        dataset_iterator = iter(sentence_dataset)
        chunk_batches = chunk_size // (model_batch_size * max_length)
        batches_to_skip = skip_chunks * chunk_batches

        if shuffle_seed is not None:
            torch.manual_seed(shuffle_seed)

        dataloader = DataLoader(
            sentence_dataset,
            batch_size=model_batch_size,
            shuffle=shuffle_seed is not None,
        )

        dataloader_iter = iter(dataloader)

        for _ in range(batches_to_skip):
            dataloader_iter.__next__()
        
        # configure hooks for the model
        tensor_buffer: Dict[str, Any] = {}

        hook_handles = []

        for tensor_name in tensor_names:
            tensor_buffer[tensor_name] = []

            def hook(module, output, tensor_name=tensor_name):
                if type(output) == tuple:
                    out = output[0]
                else:
                    out = output
                tensor_buffer[tensor_name].append(rearrange(out, "b l ... -> (b l) (...)").to(dtype=dtype).cpu())
                return output

            for name, module in model.named_modules():
                if name == tensor_name:
                    handle = module.register_forward_hook(hook)
                    hook_handles.append(handle)

        def reset_buffers():
            for tensor_name in tensor_names:
                tensor_buffer[tensor_name] = []

        reset_buffers()

        chunk_idx = 0

        progress_bar = tqdm(total=chunk_size * n_chunks)

        for batch_idx, batch in enumerate(dataloader_iter):
            batch = batch["input_ids"].to(device)

            _ = model(batch)

            progress_bar.update(model_batch_size)

            if batch_idx+1 % chunk_batches == 0:
                for tensor_name in tensor_names:
                    save_activation_chunk(tensor_buffer[tensor_name], chunk_idx, os.path.join(output_folder, tensor_name))
                
                n_act = batch_idx * model_batch_size * max_length
                print(f"Saved chunk {chunk_idx} of activations, total size: {n_act / 1e6:.2f}M activations")

                chunk_idx += 1
                
                reset_buffers()
                if chunk_idx >= n_chunks:
                    break
        
        # undersized final chunk
        if chunk_idx < n_chunks:
            for tensor_name in tensor_names:
                save_activation_chunk(tensor_buffer[tensor_name], chunk_idx, os.path.join(output_folder, tensor_name))
            
            n_act = batch_idx * model_batch_size * max_length
            print(f"Saved undersized chunk {chunk_idx} of activations, total size: {n_act / 1e6:.2f}M activations")

        for hook_handle in hook_handles:
            hook_handle.remove()
        

def save_activation_chunk(dataset, n_saved_chunks, dataset_folder):
    dataset_t = torch.cat(dataset, dim=0).to("cpu")
    os.makedirs(dataset_folder, exist_ok=True)
    with open(dataset_folder + "/" + str(n_saved_chunks) + ".pt", "wb") as f:
        torch.save(dataset_t, f)

def setup_data_new(
    model_name: str,
    dataset_name: str,
    output_folder: str,
    tensor_names: List[str],
    chunk_size: int,
    n_chunks: int,
    skip_chunks: int = 0,
    device: Optional[torch.device] = torch.device("cuda:0"),
    max_length: int = 2048,
    model_batch_size: int = 4,
    precision: Literal["float16", "float32"] = "float16",
    shuffle_seed: Optional[int] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device=device)

    # weak upper bound on number of lines
    max_lines = int((chunk_size * (n_chunks + skip_chunks)) / max_length) * 2

    print(f"Processing first {max_lines} lines of dataset...")

    sentence_dataset = make_sentence_dataset_with_Nocot(dataset_name, max_lines=max_lines)
    tokenized_sentence_dataset = simple_tokenize_per_sample(sentence_dataset, tokenizer, max_length=max_length)
    make_activation_dataset_hf(
        tokenized_sentence_dataset,
        model,
        tensor_names,
        chunk_size,
        n_chunks,
        output_folder=output_folder,
        skip_chunks=skip_chunks,
        device=device,
        max_length=max_length,
        model_batch_size=model_batch_size,
        precision=precision,
        shuffle_seed=shuffle_seed,
    )

def setup_data(
    tokenizer,
    model,
    dataset_name: str,  # Name of dataset to load
    dataset_folder: Union[str, List[str]],  # Folder to save activations to
    layer: Union[int, List[int]] = 2,
    layer_loc: str = "residual",
    start_line: int = 0,
    n_chunks: int = 100,
    chunk_size_gb: float = 2,
    skip_chunks: int = 0,
    device: torch.device = torch.device("cuda:0"),
    center_dataset: bool = False,
):
    layers = [layer] if isinstance(layer, int) else layer

    sentence_len_lower = 1000
    activation_width = get_activation_size(model.cfg.model_name, layer_loc)
    baukit = check_use_baukit(model.cfg.model_name)
    max_lines = int((chunk_size_gb * 1e9 * n_chunks) / (activation_width * sentence_len_lower * 2))
    print(f"Setting max_lines to {max_lines} to minimize sentences processed")

    sentence_dataset = make_sentence_dataset_with_Nocot(dataset_name, max_lines=max_lines, start_line=start_line)
    tensor_names = [make_tensor_name(layer, layer_loc, model.cfg.model_name) for layer in layers]
    tokenized_sentence_dataset = simple_tokenize_per_sample(sentence_dataset, tokenizer, max_length=MAX_SENTENCE_LEN)
    token_loader = DataLoader(tokenized_sentence_dataset, batch_size=MODEL_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    if baukit:
        assert type(dataset_folder) == str, "Baukit only supports single dataset folder"
        make_activation_dataset(
            sentence_dataset=token_loader,
            model=model,
            tensor_name=tensor_names[0],
            activation_width=activation_width,
            baukit=baukit,
            dataset_folder=dataset_folder,
            chunk_size_gb=chunk_size_gb,
            device=device,
            layer=layers[0],
            n_chunks=n_chunks,
            max_length=MAX_SENTENCE_LEN,
            model_batch_size=MODEL_BATCH_SIZE,
            center_dataset=center_dataset,
        )
    else:
        dataset_folder = [dataset_folder] if isinstance(dataset_folder, str) else dataset_folder
        n_datapoints = make_activation_dataset_tl(
            sentence_dataset=token_loader,
            model=model,
            activation_width=activation_width,
            dataset_folders=dataset_folder,
            chunk_size_gb=chunk_size_gb,
            device=device,
            layers=[layer] if isinstance(layer, int) else layer,
            tensor_loc=layer_loc,
            n_chunks=n_chunks,
            max_length=MAX_SENTENCE_LEN,
            model_batch_size=MODEL_BATCH_SIZE,
            skip_chunks=skip_chunks,
            center_dataset=center_dataset
        )
        return n_datapoints

def simple_tokenize_per_sample(dataset, tokenizer, max_length=256):
    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {"input_ids": tokens["input_ids"].squeeze(0)}

    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

def collate_fn(batch):
    return {
        "input_ids": torch.stack([torch.tensor(example["input_ids"]) for example in batch])
    }



def setup_token_data(cfg, tokenizer, model):
    # sentence_dataset = make_sentence_dataset(cfg.dataset_name)
    sentence_dataset = make_sentence_dataset_with_Nocot(cfg.dataset_name)
    tokenized_sentence_dataset = simple_tokenize_per_sample(sentence_dataset, tokenizer, max_length=cfg.max_length)
    token_loader = DataLoader(tokenized_sentence_dataset, batch_size=cfg.model_batch_size, shuffle=True, collate_fn=collate_fn)
    return token_loader
