import os
import re
import argparse
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns


def extract_scores(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    score = float(re.search(r"Score: ([\-\d\.]+)", content).group(1))
    top_only_score = float(re.search(r"Top only score: ([\-\d\.]+)", content).group(1))
    random_only_score = float(
        re.search(r"Random only score: ([\-\d\.]+)", content).group(1)
    )
    return score, top_only_score, random_only_score


def collect_data(root_dir):
    scores = []
    top_scores = []
    random_scores = []
    for i in range(50):
        filepath = os.path.join(root_dir, f"feature_{i}", "explanation.txt")
        if not os.path.isfile(filepath):
            print(f"⚠️ Warning: Missing file {filepath}, skipped.")
            continue
        s, top, rnd = extract_scores(filepath)
        scores.append(s)
        top_scores.append(top)
        random_scores.append(rnd)
    return scores, top_scores, random_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cotdir", required=True, help="Path to CoT feature directory")
    parser.add_argument(
        "--nocotdir", required=True, help="Path to NoCoT feature directory"
    )
    args = parser.parse_args()

    cot_scores, cot_top, cot_rand = collect_data(args.cotdir)
    nocot_scores, nocot_top, nocot_rand = collect_data(args.nocotdir)

    print("\n--- Explanation Score Comparison ---")
    print("CoT mean:", np.mean(cot_scores), "std:", np.std(cot_scores))
    print("NoCoT mean:", np.mean(nocot_scores), "std:", np.std(nocot_scores))

    t_stat, p_val = ttest_ind(cot_scores, nocot_scores, equal_var=False)
    print("T-test: t =", round(t_stat, 4), ", p =", round(p_val, 4))

    # sns.set_theme(style="whitegrid", font_scale=1.4)
    # plt.rcParams.update({'axes.titlepad': 10})

    # plt.figure(figsize=(8, 6))
    # sns.boxplot(data=[cot_scores, nocot_scores], palette="Set2")
    # plt.xticks([0, 1], ["CoT", "NoCoT"], fontsize=14)
    # plt.ylabel("Explanation Score", fontsize=16)
    # plt.title("Comparison of Feature Explanation Scores", fontsize=18)
    # plt.tight_layout()
    # plt.savefig("score_comparison_2.8b_0.0027_new_.png", dpi=300)
    # print("Saved plot successfully")

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 9))

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 24,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )

    sns.boxplot(data=[cot_scores, nocot_scores], palette="Set2")

    plt.ylabel("Explanation Score")
    plt.title("Comparison of Feature Explanation Scores")

    ax = plt.gca()
    ax.set_xticklabels(["CoT", "NoCoT"], fontweight="bold", fontsize=18)

    for label in ax.get_yticklabels():
        label.set_fontsize(18)
        label.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("score_comparison_2.8b_0.0027_final.png", dpi=600)
    plt.show()

    # plt.figure(figsize=(6, 4))
    # box_data = [cot_scores, nocot_scores]
    # box_colors = ['tab:orange', 'tab:blue']

    # bp = plt.boxplot(box_data, patch_artist=True, labels=['CoT', 'NoCoT'])

    # for patch, color in zip(bp['boxes'], box_colors):
    #     patch.set_facecolor(color)
    #     patch.set_alpha(0.7)

    # plt.ylabel("Explanation Score")
    # plt.title("Comparison of Feature Explanation Scores")
    # plt.tight_layout()
    # plt.savefig("score_comparison_70m_0.0001.png", dpi=300)
    # print("Saved plot successfully")
    # plt.show()
