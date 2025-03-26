import os
import h5py
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ranksums, kruskal
plt.figure(figsize=(5, 4))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


exp_names = {"01-frostbite-nofa-01-24-25": "No Factorization",
             "02-frostbite-lora-01-24-25": "Low-Rank Factorization",
             "03-frostbite-nofa-small-01-24-25": "No Factorization (Small)"}

exp_dfs = dict()

for exp in exp_names.keys():
    data_list = []
    subdirs = [d for d in os.listdir(exp) if os.path.isdir(os.path.join(exp, d))]

    for subdir in subdirs:
        h5_path = os.path.join(exp, subdir, "statistics.h5")
        if not os.path.isfile(h5_path):
            continue

        with h5py.File(h5_path, "r") as f:
            if "iter" not in f:
                continue
            generations = sorted(f["iter"].keys(), key=lambda x: int(x))#[0:99]
            values = []
            for gen in generations:
                # if gen == "100":
                #     break
                vals = f[f"iter/{gen}/InteractionDist/max"][()]  # read dataset
                values.append(vals)
            # Store each subdir's series with numeric index
            data_list.append(pd.Series(values, index=[int(g) for g in generations]))

    if not data_list:
        continue

    # Combine all subdirectory series into one DataFrame
    data_df = pd.concat(data_list, axis=1)
    exp_dfs[exp] = data_df
    mean_vals = data_df.mean(axis=1)
    sem_vals  = data_df.sem(axis=1)
    ci95      = 1.96 * sem_vals

    x = mean_vals.index  # generation numbers
    plt.plot(x, mean_vals, label=exp_names[exp])
    plt.fill_between(x, mean_vals - ci95, mean_vals + ci95, alpha=0.2)

plt.xlabel("Generation")
plt.ylabel("Score")
plt.title("Frostbite")
plt.legend()
plt.tight_layout()
plt.savefig("media/frostbite-reward.png")
plt.savefig("media/frostbite-reward.pdf")

print(exp_dfs)
nofa = exp_dfs["01-frostbite-nofa-01-24-25"]
fa = exp_dfs["02-frostbite-lora-01-24-25"]
nofa_small = exp_dfs["03-frostbite-nofa-small-01-24-25"]
nofa_300 = nofa.loc[300].dropna().values
fa_300 = fa.loc[300].values
nofa_small_300 = nofa_small.loc[300].dropna().values
print("Computing p value between all three methods at generation 300")
stat, p = kruskal(nofa_300, fa_300, nofa_small_300)
print(f"Kruskal-Wallis p={p:.4f}")
stat, p = ranksums(nofa_300, fa_300)
glass_delta = (np.mean(fa_300) - np.mean(nofa_300)) / np.std(nofa_300)
print(f"Factorized vs No Factorization p={p:.4f}, Glass's delta={glass_delta:.4f}")
stat, p = ranksums(nofa_300, nofa_small_300)
print(f"No Factorization vs No Factorization (Small) p={p:.4f}")
stat, p = ranksums(fa_300, nofa_small_300)
glass_delta = (np.mean(fa_300) - np.mean(nofa_small_300)) / np.std(nofa_small_300)
print(f"Factorized vs No Factorization (Small) p={p:.4f}", f"Glass's delta={glass_delta:.4f}")
#
