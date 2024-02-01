# %%
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %%
BASE_PATH = "/maps/ys611/ai-refined-rtm/saved/log/AE_RTM_corr/0124_000330/"
FILE_PATH = os.path.join(BASE_PATH, 'info.log')
SAVE_PATH = os.path.join(
    '/maps/ys611/ai-refined-rtm/saved/log/AE_RTM_corr/0124_000330/', 'log_analysis')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
counts = {}
losses = {}
val_losses = {}
# Open the file
with open(FILE_PATH, 'r') as file:
    # Loop through each line in the file
    epoch = 1
    for line in file:
        if epoch not in counts:
            counts[epoch] = 0
        if 'stablize count' in line:
            counts[epoch] += 1
        # if 'stablize count' in line:
        #     counts[epoch]  = 0 if epoch not in counts else counts[epoch]+1
        if 'loss' in line and 'epoch' not in line and 'val_loss' not in line:
            losses[epoch] = float(line.split(
                'loss')[-1].split(':')[-1].split('\n')[0])
        if 'val_loss' in line:
            val_losses[epoch] = float(line.split(
                'val_loss')[-1].split(':')[-1].split('\n')[0])
            epoch += 1
# drop the last epoch
counts_per_epoch = counts.pop(epoch)

print('Done')
# %%
# Line plot of loss
# set figure size
plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid")
df = pd.DataFrame(losses.items(), columns=['epoch', 'train_loss'])
sns.lineplot(data=df, x="epoch", y="train_loss", label="Training")
df = pd.DataFrame(val_losses.items(), columns=['epoch', 'val_loss'])
sns.lineplot(data=df, x="epoch", y="val_loss", label="Validation")
# font size of x and y ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# font size of x and y labels
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('MSE Loss', fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'train_val_loss.png'), dpi=200)
plt.show()

# %%
# Line plot of stablize count
# set figure size
plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid")
df = pd.DataFrame(counts_per_epoch.items(), columns=['epoch', 'count'])
sns.lineplot(data=df, x="epoch", y="count")
# font size of x and y ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# font size of x and y labels
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Times of Stabilization', fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'stabilize_count.png'), dpi=200)
plt.show()

# %%
