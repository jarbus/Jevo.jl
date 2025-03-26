import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.figure(figsize=(5, 4))

exp_names = {"01-relu-nofa-01-06-25": "No Factorization",
             "02-relu-lora-01-06-25": "Low-Rank Factorization (Larger Rank)",
             "03-relu-nofa-small-01-06-25": "No Factorization (Small)",
             "07-relu-rank1-lora-03-21-25": "Low-Rank Factorization (Smaller Rank)"
             }

for exp in exp_names.keys():
    data_list = []
    # Collect all subdirectories
    subdirs = [d for d in os.listdir(exp) if os.path.isdir(os.path.join(exp, d))]
    for subdir in subdirs:
        print(exp, subdir)
        csv_path = os.path.join(exp, subdir, "dist.csv")
        df = pd.read_csv(csv_path, header=None)
        data_list.append(df.iloc[:, 4])  # 5th column

    data = pd.concat(data_list, axis=1)
    mean_vals = data.mean(axis=1)
    sem_vals = data.sem(axis=1)  # standard error of the mean
    ci95 = 1.96 * sem_vals  # ~95% CI

    x = range(len(mean_vals))
    plt.plot(x, mean_vals, label=exp_names[exp])
    plt.fill_between(x, mean_vals - ci95, mean_vals + ci95, alpha=0.2)

plt.xlabel("Generation")
plt.ylabel("Negative Cross-Entropy Loss")
plt.title("Training Loss on TinyStories Dataset")
plt.legend()
plt.tight_layout()
plt.savefig("media/tfr-loss.png")
plt.savefig("media/tfr-loss.pdf")
