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

exp_names = {"01-relu-nofa-01-06-25": "No Factorization", "02-relu-lora-01-06-25": "Low-Rank Factorization", "03-relu-nofa-small-01-06-25": "No Factorization (Small)"}

exp_dfs = dict()

for exp in exp_names.keys():
    data_list = []
    subdirs = [d for d in os.listdir(exp) if os.path.isdir(os.path.join(exp, d))]

    for subdir in subdirs:
        csv_path = os.path.join(exp, subdir, "dist.csv")
        if not os.path.exists(csv_path):
            continue
        print(f"Reading {csv_path}")
        df = pd.read_csv(csv_path, header=None)
        data_list.append(df.iloc[:, 4])  # 5th column

    if not data_list:
        continue

    # Combine all subdirectory series into one DataFrame
    data_df = pd.concat(data_list, axis=1)
    exp_dfs[exp] = data_df

print(exp_dfs)
nofa = exp_dfs["01-relu-nofa-01-06-25"]
fa = exp_dfs["02-relu-lora-01-06-25"]
nofa_small = exp_dfs["03-relu-nofa-small-01-06-25"]
nofa_299 = nofa.loc[299].dropna().values
fa_299 = fa.loc[299].values
nofa_small_299 = nofa_small.loc[299].dropna().values
print("Computing p value between all three methods at generation 299")
stat, p = kruskal(nofa_299, fa_299, nofa_small_299)
print(f"Kruskal-Wallis p={p:.4f}")
stat, p = ranksums(nofa_299, fa_299)
glass_delta = (np.mean(fa_299) - np.mean(nofa_299)) / np.std(nofa_299)
print(f"Factorized vs No Factorization p={p:.4f}, Glass's delta={glass_delta:.4f}")
stat, p = ranksums(nofa_299, nofa_small_299)
print(f"No Factorization vs No Factorization (Small) p={p:.4f}")
stat, p = ranksums(fa_299, nofa_small_299)
glass_delta = (np.mean(fa_299) - np.mean(nofa_small_299)) / np.std(nofa_small_299)
print(f"Factorized vs No Factorization (Small) p={p:.4f}", f"Glass's delta={glass_delta:.4f}")
#
