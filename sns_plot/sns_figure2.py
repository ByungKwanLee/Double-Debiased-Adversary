import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

labels = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

vgg_trades = np.array([[0.562, 0.693, 0.309, 0.179, 0.257, 0.404, 0.525, 0.58 , 0.679, 0.598]])
vgg_mart = np.array([[0.576, 0.701, 0.366, 0.188, 0.285, 0.441, 0.579, 0.616, 0.686, 0.59]])
vgg_awp = np.array([[0.597, 0.745, 0.367, 0.187, 0.321, 0.47 , 0.617, 0.63 , 0.705, 0.609]])
res_trades = np.array([[0.602, 0.711, 0.344, 0.227, 0.322, 0.418, 0.556, 0.626, 0.688, 0.652]])
res_mart = np.array([[0.597, 0.726, 0.382, 0.248, 0.346, 0.422, 0.589, 0.641, 0.704, 0.634]])
res_awp = np.array([[0.633, 0.756, 0.399, 0.255, 0.355, 0.459, 0.619, 0.667, 0.766, 0.646]])

vitb_trades = np.array([[0.592, 0.786, 0.367, 0.199, 0.363, 0.458, 0.525, 0.602, 0.669, 0.579]])
vitb_mart = np.array([[0.617, 0.717, 0.407, 0.208, 0.374, 0.455, 0.56 , 0.624, 0.636, 0.569]])
vitb_awp = np.array([[0.635, 0.725, 0.434, 0.205, 0.321, 0.436, 0.581, 0.681, 0.679, 0.641]])
deitb_trades = np.array([[0.605, 0.723, 0.433, 0.219, 0.277, 0.415, 0.569, 0.647, 0.745, 0.611]])
deitb_mart = np.array([[0.604, 0.716, 0.441, 0.199, 0.296, 0.489, 0.593, 0.652, 0.725, 0.617]])
deitb_awp = np.array([[0.626, 0.731, 0.412, 0.188, 0.264, 0.508, 0.605, 0.686, 0.709, 0.623]])

cifar_cnn = np.concatenate([vgg_trades, vgg_mart, vgg_awp, res_trades, res_mart, res_awp])
cifar_df_cnn = pd.DataFrame(columns=labels, data=cifar_cnn)

cifar_tran = np.concatenate([vitb_trades, vitb_mart, vitb_awp, deitb_trades, deitb_mart, deitb_awp])
cifar_df_tran = pd.DataFrame(columns=labels, data=cifar_tran)

cifar_cnn = cifar_df_cnn.set_index([["VGG-TRADES", "VGG-MART", "VGG-AWP", "ResNet-TRADES", "ResNet-MART", "ResNet-AWP"]])
cifar_df_cnn_ = cifar_cnn.stack().reset_index()
cifar_df_cnn_.columns = ['', 'Class', 'Adversarial Accuracy']

cifar_tran = cifar_df_tran.set_index([["ViT-B-TRADES", "ViT-B-MART", "ViT-B-AWP", "DeiT-B-TRADES", "DeiT-B-MART", "DeiT-B-AWP"]])
cifar_df_tran_ = cifar_tran.stack().reset_index()
cifar_df_tran_.columns = ['', 'Class', 'Adversarial Accuracy']

matplotlib.rc_file_defaults()
sns.set_style("darkgrid", {'font.family':'serif', 'font.serif':['Times New Roman']})
deep = ["#b30f0f", "#104a77", "#a460dc", "#e44e0c", "#318500", "#00d88f",
        "#ff9595", "#2476ff", "#eb00a0", "#ffcd00", "#9ed10f", "#a7a7a7"]
pallete = {"VGG-TRADES": deep[0], "VGG-MART": deep[1], "VGG-AWP": deep[2], "ResNet-TRADES": deep[3], "ResNet-MART": deep[4], "ResNet-AWP": deep[5]}
pallete2 = {"ViT-B-TRADES": deep[6], "ViT-B-MART": deep[7], "ViT-B-AWP": deep[8], "DeiT-B-TRADES": deep[9], "DeiT-B-MART": deep[10], "DeiT-B-AWP": deep[11]}

fig = plt.figure(figsize=(6, 3), dpi=900)
b = sns.lineplot(x='Class', y='Adversarial Accuracy', hue='', data=cifar_df_cnn_, alpha=0.7, palette=pallete, marker='o', markersize=4, linewidth=1.4)
b2 = sns.lineplot(x='Class', y='Adversarial Accuracy', hue='', data=cifar_df_tran_, alpha=0.7, palette=pallete2, marker='<', markersize=4, linewidth=1.4, linestyle='-.')

leg = b.legend(loc='lower right', title='', ncol=2, fontsize=7)
leg_lines = leg.get_lines()
leg_lines[6].set_linestyle('dashdot')
leg_lines[7].set_linestyle('dashdot')
leg_lines[8].set_linestyle('dashdot')
leg_lines[9].set_linestyle('dashdot')
leg_lines[10].set_linestyle('dashdot')
leg_lines[11].set_linestyle('dashdot')

b.set(ylim=(0.13, 0.8))
b.set(xlim=(-0.05, 9.05))

b.tick_params(axis='y', labelsize=8)
b.tick_params(axis='x', labelsize=7)
b.set_xticks(labels)
b.set_xticklabels(labels, rotation=45, ha='center', rotation_mode='anchor', position=(0, -0.04))

# b.axes.set_title("Architecture", fontsize=13)
b.set_ylabel("Adversarial Robustness", fontsize=12)
b.set_xlabel("(b) AT-based Learning", fontsize=12)

plt.setp(b.get_legend().get_texts(), fontsize=5)
plt.tight_layout()
plt.savefig("./fig1b.png")
