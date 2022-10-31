import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

labels = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

vgg16_adv = np.array([[0.561, 0.677, 0.326, 0.148, 0.257, 0.405, 0.568, 0.598, 0.675, 0.568]])
vgg19_adv = np.array([[0.572, 0.684, 0.302, 0.136, 0.279, 0.385, 0.575, 0.603, 0.685, 0.588]])
res18_adv = np.array([[0.591, 0.711, 0.358, 0.22, 0.296, 0.428, 0.564, 0.636, 0.685, 0.634]])
res50_adv = np.array([[0.598, 0.733, 0.392, 0.242, 0.321, 0.425, 0.584, 0.652, 0.693, 0.655]])
wrn28_adv = np.array([[0.603, 0.733, 0.394, 0.268, 0.339, 0.424, 0.584, 0.644, 0.697, 0.66]])
wrn70_adv = np.array([[0.623, 0.732, 0.423, 0.281, 0.354, 0.458, 0.593, 0.665, 0.688, 0.689]])

vits_adv = np.array([[0.592, 0.686, 0.302, 0.148, 0.221, 0.438, 0.643, 0.667, 0.648, 0.514]])
vitb_adv = np.array([[0.614, 0.697, 0.418, 0.205, 0.34 , 0.405, 0.516, 0.661, 0.605, 0.575]])
deits_adv = np.array([[0.59 , 0.696, 0.371, 0.168, 0.301, 0.382, 0.565, 0.62 , 0.626, 0.521]])
deitb_adv = np.array([[0.559, 0.742, 0.393, 0.191, 0.322, 0.427, 0.544, 0.637, 0.629, 0.592]])
swins_adv = np.array([[0.55 , 0.663, 0.4  , 0.139, 0.303, 0.426, 0.475, 0.594, 0.725, 0.527]])
tnts_adv = np.array([[0.587, 0.706, 0.338, 0.198, 0.347, 0.392, 0.547, 0.62 , 0.663, 0.567]])

cifar_cnn = np.concatenate([vgg16_adv, vgg19_adv,res18_adv, res50_adv, wrn28_adv, wrn70_adv])
cifar_df_cnn = pd.DataFrame(columns=labels, data=cifar_cnn)

cifar_tran = np.concatenate([vits_adv, vitb_adv, deits_adv, deitb_adv, swins_adv, tnts_adv])
cifar_df_tran = pd.DataFrame(columns=labels, data=cifar_tran)

cifar_cnn = cifar_df_cnn.set_index([["VGG-16", "VGG-19", "ResNet-18", "ResNet-50", "WRN-28-10", "WRN-70-10"]])
cifar_df_cnn_ = cifar_cnn.stack().reset_index()
cifar_df_cnn_.columns = ['', 'Class', 'Adversarial Accuracy']

cifar_tran = cifar_df_tran.set_index([["ViT-S/16", "ViT-B/16", "DeiT-S/16", "DeiT-B/16", "Swin-S/4", "TNT-S/16"]])
cifar_df_tran_ = cifar_tran.stack().reset_index()
cifar_df_tran_.columns = ['', 'Class', 'Adversarial Accuracy']

matplotlib.rc_file_defaults()
sns.set_style("darkgrid", {'font.family':'serif', 'font.serif':['Times New Roman']})
# deep = ["#f0be39", "#31a354", "#1f77b4", "#d70217", "#7cccec", "#b96528", "#d4acb3", "#34bc4c", "#fdb955", "#e6544e"]
deep = ["#b30f0f", "#104a77", "#a460dc", "#e44e0c", "#318500", "#00d88f",
        "#ff9595", "#2476ff", "#eb00a0", "#ffcd00", "#9ed10f", "#a7a7a7"]

pallete = {"VGG-16": deep[0], "VGG-19": deep[1],"ResNet-18": deep[2], "ResNet-50": deep[3], "WRN-28-10": deep[4], "WRN-70-10": deep[5]}
pallete2 = {"ViT-S/16": deep[6], "ViT-B/16": deep[7], "DeiT-S/16": deep[8], "DeiT-B/16": deep[9], "Swin-S/4": deep[10], "TNT-S/16": deep[11]}

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
b.set_xlabel("(a) Network Architecture", fontsize=12)

plt.setp(b.get_legend().get_texts(), fontsize=5)
plt.tight_layout()
plt.savefig("./fig1a.png")
