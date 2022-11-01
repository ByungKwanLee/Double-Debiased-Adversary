import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

content = np.array(['VGG-16', 'ResNet-18', 'ViT-B/16', 'DeiT-B/16'])

# tiny-10
top_k = np.array([[0.46633333, 0.5316667, 0.6009999, 0.60800004]])
top_k_daml = np.array([[0.46633333, 0.5316667, 0.6009999, 0.60800004]])
bot_k = np.array([[0.106999986, 0.14633335, 0.1846667, 0.20133331]])
bot_k_daml = np.array([[0.106999986, 0.14633335, 0.1846667, 0.20133331]])

top = np.concatenate([top_k, top_k_daml])
tiny_df_top = pd.DataFrame(columns=content, data=top)
bottom = np.concatenate([bot_k, bot_k_daml])
tiny_df_bot = pd.DataFrame(columns=content, data=bottom)

tiny_t = tiny_df_top.set_index([["top-k/AT", "top-k/DAML"]])
tiny_df_top_ = tiny_t.stack().reset_index()
tiny_df_top_.columns = ['', 'Network', 'Adversarial Accuracy']

tiny_b = tiny_df_bot.set_index([["bottom-k/AT", "bottom-k/DAML"]])
tiny_df_bot_ = tiny_b.stack().reset_index()
tiny_df_bot_.columns = ['', 'Network', 'Adversarial Accuracy']

sns.set_style("darkgrid", {'font.family':'serif', 'font.serif':['Times New Roman']})
# deep = ["#f0be39", "#31a354", "#1f77b4", "#d70217", "#7cccec", "#b96528", "#d4acb3", "#34bc4c", "#fdb955", "#e6544e"]
deep = ["#057ff2", "#f20560", "#2105f2", "#f20505"]

pallete = {"top-k/AT": deep[0], "bottom-k/AT": deep[1], "top-k/DAML": deep[2], "bottom-k/DAML": deep[3]}

fig = plt.figure(figsize=(4, 2.8), dpi=900)
b = sns.barplot(x='Network', y='Adversarial Accuracy', hue='', data=tiny_df_top_, alpha=0.9, palette='Blues_d')
b = sns.barplot(x='Network', y='Adversarial Accuracy', hue='', data=tiny_df_bot_, alpha=0.9, palette='Reds_d')

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
plt.savefig("./fig_disparity_tiny.png")
