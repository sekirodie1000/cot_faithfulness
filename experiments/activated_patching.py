import re, os, sys, argparse, pickle, json, math, random, gc
from glob import glob
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_dataset
from unicodedata import normalize

import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer

sys.path.append(os.path.abspath(".."))
from autoencoders.sae_ensemble import FunctionalTiedSAE
from activation_dataset import make_sentence_dataset_with_Nocot


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="EleutherAI/pythia-2.8b")
parser.add_argument(
    "--layer", type=int, default=0, help="Layer number for patching (residual/MLP)"
)
parser.add_argument("--layer_loc", default="resid")
parser.add_argument("--rank", type=int, default=0, help="SAE rank index (0–8)")
parser.add_argument("--dict_nocot", required=True)
parser.add_argument("--dict_cot", required=True)
parser.add_argument("--acts_nocot_dir", required=True)
parser.add_argument("--acts_cot_dir", required=True)
parser.add_argument("--topk", type=int, default=20, help="Top-k features to patch")
parser.add_argument(
    "--max_samples", type=int, default=2000, help="Max number of samples to evaluate"
)
parser.add_argument("--out", default="ll_gain_curve.png")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)


def build_gsm8k_lookup(split="train"):
    ds = load_dataset("openai/gsm8k", split=split, verification_mode="no_checks")
    table = {}
    for ex in ds:
        q_txt = normalize("NFKC", ex["question"].strip())
        ans = ex["answer"].split("####")[-1].strip()
        table[q_txt] = ans
    return table


GSM_LOOKUP = build_gsm8k_lookup()
print("lookup size =", len(GSM_LOOKUP))


# --- 1. Load model ---
model: HookedTransformer = HookedTransformer.from_pretrained(
    args.model,
    device=device,
    torch_dtype=torch.bfloat16,
    fold_ln=False,
    center_writing_weights=False,
)
tokenizer = model.tokenizer
d_model = model.cfg.d_model
print(f"Loaded model {args.model}  (d_model={d_model})")


# --- 2. Load SAE dictionaries ---
def load_dict(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


dict_nocot = load_dict(args.dict_nocot)
dict_cot = load_dict(args.dict_cot)

sae_nocot: FunctionalTiedSAE = dict_nocot[args.layer][args.rank][0]
sae_cot: FunctionalTiedSAE = dict_cot[args.layer][args.rank][0]
sae_nocot.to_device(device)
sae_cot.to_device(device)
print(f"SAE (layer={args.layer}, rank={args.rank}) with dict_size={sae_nocot.n_feats}")

# --- 3. Load GSM8K test set ---
gsm = make_sentence_dataset_with_Nocot("openai/gsm8k")


def extract_answer_str(example):
    return example["answer"].split("####")[-1].strip()


def extract_question_from_prompt(decoded: str) -> str:
    try:
        q = decoded.split("Q:")[1].split("\nA:")[0]
    except Exception:
        q = decoded
    return normalize("NFKC", q.strip())


# --- 4. Utility: compute log-prob of answer tokens ---
def answer_logprob(model, prompt_ids, answer, hooks=None):
    ans_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    full_ids = torch.cat(
        [prompt_ids.squeeze(0), torch.tensor(ans_ids, device=prompt_ids.device)], 0
    )
    with model.hooks(hooks or []):
        logits = model(full_ids[:-1].unsqueeze(0), return_type="logits")
    logprobs = F.log_softmax(logits.float(), dim=-1)
    tgt = full_ids[1:].unsqueeze(0)
    return logprobs.gather(2, tgt.unsqueeze(-1)).squeeze(-1).sum().item()


# --- 5. Gather activation and input_ids files ---


def sorted_pt_files(
    folder: str, *, numeric_only: bool = True, prefix: str = "", layer_tag: str = ""
) -> List[str]:
    paths = []
    for fname in os.listdir(folder):
        if not fname.endswith(".pt"):
            continue
        if prefix and not fname.startswith(prefix):
            continue
        if layer_tag and layer_tag not in fname:
            continue
        if numeric_only and not re.fullmatch(r"\d+\.pt", fname):
            continue
        paths.append(os.path.join(folder, fname))

    paths.sort(key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    return paths


acts_files_nocot = sorted_pt_files(args.acts_nocot_dir, numeric_only=True)
acts_files_cot = sorted_pt_files(args.acts_cot_dir, numeric_only=True)
input_ids_files = sorted_pt_files(
    args.acts_nocot_dir, numeric_only=False, prefix="input_ids_layer", layer_tag="chunk"
)


assert (
    len(acts_files_nocot) == len(acts_files_cot) == len(input_ids_files)
), f"Chunk counts: NoCoT {len(acts_files_nocot)}, CoT {len(acts_files_cot)}, input_ids {len(input_ids_files)}"


# --- 6. Evaluation loop over samples ---
K = args.topk
ll_gain_list = []
pbar = tqdm(total=min(args.max_samples, len(gsm)), desc="patch-eval")
sample_cnt = 0

for chunk_idx, (act_f_nc, act_f_c, ids_f) in enumerate(
    zip(acts_files_nocot, acts_files_cot, input_ids_files)
):
    if sample_cnt >= args.max_samples:
        break

    acts_nc = torch.load(act_f_nc, map_location=device).to(torch.bfloat16)
    acts_c = torch.load(act_f_c, map_location=device).to(torch.bfloat16)
    ids_list: List[torch.Tensor] = torch.load(ids_f)

    idx_start = 0
    for local_idx, ids in enumerate(ids_list):
        if sample_cnt >= args.max_samples:
            break

        idx_end = idx_start + ids.shape[0] - 1
        act_nc_last = acts_nc[idx_end].unsqueeze(0).to(device)
        act_c_last = acts_c[idx_end].unsqueeze(0).to(device)
        dtype = sae_nocot.encoder.dtype
        act_nc_last = act_nc_last.to(dtype)
        act_c_last = act_c_last.to(dtype)
        prompt_ids = ids.unsqueeze(0).to(device)

        code_nc = sae_nocot.encode(act_nc_last)
        code_c = sae_cot.encode(act_c_last)
        diff = (code_c - code_nc).abs().squeeze(0)
        topk_idx = diff.topk(k=K).indices

        seq_len = ids.numel()
        if sample_cnt < 3:
            print("seq_len =", seq_len)

        # =============  DEBUG 1  =================
        if sample_cnt < 10:
            print(f"\nSample {sample_cnt}")
            print("top‑k diff:", diff[topk_idx][:10].cpu().tolist())
        # =========================================

        patched_code = code_nc.clone()
        patched_code[0, topk_idx] = code_c[0, topk_idx]
        patched_act = sae_nocot.decode(patched_code).squeeze(0)

        decoded_prompt = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
        q_txt = extract_question_from_prompt(decoded_prompt)
        gt_answer = GSM_LOOKUP.get(q_txt, None)

        if gt_answer is None:
            if sample_cnt < 5:
                print("⚠️ Question does not match, skip:", q_txt[:80])
            idx_start = idx_end + 1
            continue

        baseline_ll = answer_logprob(model, prompt_ids, gt_answer)

        def hook_fn(resid_pre, hook):
            resid_pre[:, -1, :] = patched_act.to(resid_pre.device)
            return resid_pre

        hooks = [(f"blocks.{args.layer}.hook_{args.layer_loc}_pre", hook_fn)]
        patched_ll = answer_logprob(model, prompt_ids, gt_answer, hooks=hooks)

        # =============  DEBUG 2  =================
        if sample_cnt < 10:
            print(
                f"baseline_ll = {baseline_ll:.3f} , patched_ll = {patched_ll:.3f} , Δ = {patched_ll - baseline_ll:.3f}"
            )
            print("question:", tokenizer.decode(prompt_ids[0]))
            print("question:", q_txt)
            print("extracted answer:", gt_answer)
        # =========================================

        ll_gain_list.append(patched_ll - baseline_ll)
        pbar.set_postfix(ll_gain=ll_gain_list[-1])
        pbar.update(1)
        sample_cnt += 1
        idx_start = idx_end + 1

    del acts_nc, acts_c
    torch.cuda.empty_cache()

pbar.close()
print(
    f"\nCompleted {sample_cnt} samples. Average Δlog-prob = {sum(ll_gain_list)/len(ll_gain_list):.3f}"
)

# --- 7. Plot and save results ---
plt.figure(figsize=(6, 4))
plt.hist(ll_gain_list, bins=40, alpha=0.75)
plt.xlabel("patched log-prob − baseline log-prob  ( >0 means patch helped )")
plt.ylabel("count")
plt.title(f"Pythia CoT→NoCoT patching  (K={K}, samples={sample_cnt})")
plt.tight_layout()
plt.savefig(args.out)
print(f"Saved plot → {args.out}")
