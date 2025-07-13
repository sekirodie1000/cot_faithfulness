import os, sys, argparse, pickle, torch, matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm
from einops import rearrange
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--sae_pkl", required=True)
p.add_argument("--layer", type=int, default=1)
p.add_argument("--layer_loc", default="resid")
p.add_argument("--top_feats", type=int, default=10)
p.add_argument("--top_tokens", type=int, default=10)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--token_amount", type=int, default=25)
p.add_argument("--out", default="boxplot.png")
args = p.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sns.set_theme(style="whitegrid", font_scale=1.3)

sys.path.append(os.path.abspath(".."))
from autoencoders.sae_ensemble import FunctionalTiedSAE


def load_sae(path, layer, rank=0):
    with open(path, "rb") as f:
        d = pickle.load(f)
    sae: FunctionalTiedSAE = d[layer][rank][0]
    sae.to_device("cpu")
    return sae


print(">> load model")
model = HookedTransformer.from_pretrained(
    args.model,
    device=DEVICE,
    torch_dtype=torch.bfloat16,
    fold_ln=False,
    center_writing_weights=False,
)

print(">> load SAE")
sae = load_sae(args.sae_pkl, layer=args.layer)

print(">> dataset")
raw = load_dataset("openai/gsm8k", split="train")
ds = (
    raw.map(lambda x: model.tokenizer(x["question"]), batched=True)
    .filter(lambda x: len(x["input_ids"]) > args.token_amount)
    .map(lambda x: {"input_ids": x["input_ids"][: args.token_amount]})
    .with_format("torch")
)
dl = DataLoader(ds["input_ids"], batch_size=args.batch_size, shuffle=False)

dict_size = sae.encoder.shape[0]
energy = torch.zeros(dict_size, dtype=torch.float32)

for ids in tqdm(dl, desc="pass-1"):
    acts = model.run_with_cache(ids.to(DEVICE))[1][
        f"blocks.{args.layer}.hook_{args.layer_loc}_post"
    ]
    acts = rearrange(acts, "b s d -> (b s) d").to("cpu", sae.encoder.dtype)
    energy += sae.encode(acts).sum(0).float()

top_feat_idx = torch.topk(energy, args.top_feats).indices.tolist()

top_scores = {f: torch.empty(0, dtype=sae.encoder.dtype) for f in top_feat_idx}

for ids in tqdm(dl, desc="pass-2"):
    acts = model.run_with_cache(ids.to(DEVICE))[1][
        f"blocks.{args.layer}.hook_{args.layer_loc}_post"
    ]
    acts = rearrange(acts, "b s d -> (b s) d").to("cpu", sae.encoder.dtype)
    codes = sae.encode(acts)
    for f in top_feat_idx:
        scores = torch.cat([top_scores[f], codes[:, f]])
        top_scores[f] = torch.topk(scores, args.top_tokens).values

feat_threshold = {f: top_scores[f][-1].item() for f in top_feat_idx}

thresholds = [round(i / 10, 1) for i in range(11)]
stats = [[] for _ in thresholds]

for ids in tqdm(dl, desc="pass-3"):
    acts = model.run_with_cache(ids.to(DEVICE))[1][
        f"blocks.{args.layer}.hook_{args.layer_loc}_post"
    ]
    acts = rearrange(acts, "b s d -> (b s) d").to("cpu", sae.encoder.dtype)
    codes = sae.encode(acts)[:, top_feat_idx]

    trigger = codes >= torch.tensor([feat_threshold[f] for f in top_feat_idx])

    for j, thr in enumerate(thresholds):
        mask_token = trigger.any(1)
        if mask_token.any():
            act_tok = acts[mask_token]
            stats[j].extend((act_tok > thr).sum(1).tolist())

    del acts, codes, trigger

for j in range(len(stats)):
    s = stats[j]
    if len(s) < args.top_feats:
        s += [0] * (args.top_feats - len(s))
    stats[j] = s[: args.top_feats]


TITLE = "Activated Neurons per SAE Feature"

plt.figure(figsize=(6, 4))
plt.boxplot(
    stats,
    positions=range(len(thresholds)),
    patch_artist=True,
    boxprops=dict(facecolor=sns.color_palette("Set2")[0], alpha=0.6),
    medianprops=dict(color="k"),
)
plt.xticks(range(len(thresholds)), thresholds)
plt.xlabel("Threshold")
plt.ylabel("# neurons activated")
plt.title(TITLE, pad=8)

plt.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.12)

plt.savefig(args.out, dpi=300, bbox_inches="tight")
print(">> plot saved to", args.out)
