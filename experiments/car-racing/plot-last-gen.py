import json
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ranksums
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib
# plt.figure(figsize=(5, 4))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

nofa_str = '01-nofa-01-24-25'
fa_str = '02-lora-01-24-25'
nofa_small_str = '03-nofa-small-01-24-25/'

waves = []
with open("./last-gen-information.json", "r") as f:
    for line in f:
        waves.extend(json.loads(line))


# Convert to a DataFrame
df = pd.DataFrame(waves)

# multiply all 'mean' entries x3 where WaveName==SecondWave
# we do this because we sum up all points over 6 evaluations and divide by 6
# but the first wave has only run two evaluations, so we need to cancel out the /6 by multiplying by 3
df.loc[df['wave_name'] == 'FirstWave', 'mean'] *= 3

# Compute p values

print("Comparing Factorized and Non-Factorized Second Wave time p value")
nofa_time = df[(df['experiment_name'] == nofa_str) & (df['wave_name']=='SecondWave')]['time_taken']
fa_time = df[(df['experiment_name'] == fa_str) & (df['wave_name']=='SecondWave')]['time_taken']
stat, p = ranksums(nofa_time, fa_time)
glass_delta = (np.mean(fa_time) - np.mean(nofa_time)) / np.std(nofa_time)
print(f"Mean score: p={p:.4f}, Glass's delta={glass_delta:.4f}")
print("Comparing Factorized and Non-Factorized Second Wave score p value")
nofa_score = df[(df['experiment_name'] == nofa_str) & (df['wave_name']=='SecondWave')]['mean']
fa_score = df[(df['experiment_name'] == fa_str) & (df['wave_name']=='SecondWave')]['mean']
stat, p = ranksums(nofa_score, fa_score)
glass_delta = (np.mean(fa_score) - np.mean(nofa_score)) / np.std(nofa_score)
print(f"Mean score: p={p:.4f}, Glass's delta={glass_delta:.4f}")
print("Computing Factorized and Non-Factorized first wave time taken p value")
nofa_time = df[(df['experiment_name'] == nofa_str) & (df['wave_name']=='FirstWave')]['time_taken']
fa_time = df[(df['experiment_name'] == fa_str) & (df['wave_name']=='FirstWave')]['time_taken']
glass_delta = (np.mean(fa_time) - np.mean(nofa_time)) / np.std(nofa_time)
stat, p = ranksums(nofa_time, fa_time)
print(f"Time taken: p={p:.4f}, Glass's delta={glass_delta:.4f}")

print("Computing Factorized and Non-Factorized first wave mean score p value")
nofa_score = df[(df['experiment_name'] == nofa_str) & (df['wave_name']=='FirstWave')]['mean']
fa_score = df[(df['experiment_name'] == fa_str) & (df['wave_name']=='FirstWave')]['mean']
glass_delta = (np.mean(fa_score) - np.mean(nofa_score)) / np.std(nofa_score)
stat, p = ranksums(nofa_score, fa_score)
print(f"Time taken: p={p:.4f}, Glass's delta={glass_delta:.4f}")


# Plot
df = df[df['experiment_name'].str.startswith(('01', '02'))]

df.loc[df['experiment_name']== nofa_str, 'experiment_name'] = "Non-Factorized"
df.loc[df['experiment_name']== fa_str, 'experiment_name'] = "Factorized"
df.loc[df['experiment_name']== nofa_small_str, 'experiment_name'] = "Non-Factorized, Small"



df.loc[df['wave_name']=='SecondWave', 'wave_name'] = "Second Stage"
df.loc[df['wave_name']=='FirstWave', 'wave_name'] = "First Stage"

for stage in 'First Stage', 'Second Stage':
    print(stage)
    filtered = df[df['wave_name'] == stage]
    summary = filtered.groupby('experiment_name').agg(
        mean_time_taken=('time_taken', 'mean'),
        mean_of_means=('mean', 'mean')
    )
    print(summary)

# Plot
# plt.figure(figsize=(6, 6))
sns.violinplot(data=df, x='experiment_name', y='mean', hue='wave_name', split=True, palette="colorblind", inner=None)
sns.stripplot(data=df, x='experiment_name', y='mean', hue='wave_name', size=4, jitter=0, legend=False)
plt.title("CarRacing Generation 40 Average Score\n")
plt.ylabel("Score")
plt.legend(title='Selection Stage')
plt.xlabel("")
plt.tight_layout()
plt.savefig("media/cr-last-gen-mean-score.png")
plt.savefig("media/cr-last-gen-mean-score.pdf")

plt.clf()
# plt.figure(figsize=(6, 6))
sns.violinplot(data=df, x='experiment_name', y='time_taken', hue='wave_name', split=True, palette="colorblind", inner=None)
sns.stripplot(data=df, x='experiment_name', y='time_taken', hue='wave_name', size=4, jitter=0, legend=False)
# above but disable legend


plt.title("CarRacing Generation 40 Evaluation Time\n")
plt.ylabel("Time taken (s)")
plt.legend(title='Selection Stage')
plt.xlabel("")
plt.tight_layout()
plt.savefig("media/cr-last-gen-mean-time.png")
plt.savefig("media/cr-last-gen-mean-time.pdf")
