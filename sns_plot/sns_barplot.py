import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

content = np.array(['VGG-16', 'ResNet-18', 'ViT-B/16', 'DeiT-B/16'])

# CIFAR-10
top_k = np.array([[0.652, 0.678, 0.6573334, 0.6696667]])
top_k_daml = np.array([[0.652, 0.678, 0.6573334, 0.6696667]])
bot_k = np.array([[0.24266666, 0.29466668, 0.31633332, 0.30266666]])
bot_k_daml = np.array([[0.24266666, 0.29466668, 0.31633332, 0.30266666]])

top = np.concatenate([top_k, top_k_daml])
cifar_df_top = pd.DataFrame(columns=content, data=top)
bottom = np.concatenate([bot_k, bot_k_daml])
cifar_df_bot = pd.DataFrame(columns=content, data=bottom)

cifar_t = cifar_df_top.set_index([["top-k/AT", "top-k/DAML"]])
cifar_df_top_ = cifar_t.stack().reset_index()
cifar_df_top_.columns = ['', 'Network', 'Adversarial Accuracy']

cifar_b = cifar_df_bot.set_index([["bottom-k/AT", "bottom-k/DAML"]])
cifar_df_bot_ = cifar_b.stack().reset_index()
cifar_df_bot_.columns = ['', 'Network', 'Adversarial Accuracy']

sns.set_style("darkgrid", {'font.family':'serif', 'font.serif':['Times New Roman']})
# deep = ["#f0be39", "#31a354", "#1f77b4", "#d70217", "#7cccec", "#b96528", "#d4acb3", "#34bc4c", "#fdb955", "#e6544e"]
deep = ["#057ff2", "#f20560", "#2105f2", "#f20505"]

pallete = {"top-k/AT": deep[0], "bottom-k/AT": deep[1], "top-k/DAML": deep[2], "bottom-k/DAML": deep[3]}

fig = plt.figure(figsize=(4, 2.8), dpi=900)
b = sns.barplot(x='Network', y='Adversarial Accuracy', hue='', data=cifar_df_top_, alpha=0.9, palette='Blues_d')
b = sns.barplot(x='Network', y='Adversarial Accuracy', hue='', data=cifar_df_bot_, alpha=0.9, palette='Reds_d')

width = 0.3

for bar in b.patches:
    x = bar.get_x()
    old_width = bar.get_width()
    bar.set_width(width)
    bar.set_x(x + (old_width - width) / 2)

b.legend(loc='upper right', title='', frameon=True, ncol=2, fontsize=7)

b.set(ylim=(0.0, 0.8))

b.tick_params(axis='y', labelsize=8)
b.tick_params(axis='x', labelsize=7)
# b.set_xticks(content)
# b.set_xticklabels(content, rotation=45, ha='center', rotation_mode='anchor', position=(0, -0.04))

# b.axes.set_title("Architecture", fontsize=13)
b.set_ylabel("Adversarial Robustness", fontsize=12)
b.set(xlabel=None)

plt.setp(b.get_legend().get_texts(), fontsize=5)
plt.tight_layout()
plt.savefig("./fig_disparity_cifar.png")
