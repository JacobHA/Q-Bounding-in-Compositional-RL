import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np


def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1",
              title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs):

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    plt.figure(constrained_layout=True)
    sns.set(style="whitegrid")
    # MAke the linewidth thicker
    g = sns.lineplot(data=data, x=xaxis, y=value, hue=condition)

    plt.legend(loc='best').set_draggable(True)
    # Make legend linewidth thicker
    for line in plt.gca().get_legend().get_lines():
        line.set_linewidth(4)

    # Change fontsize of legend
    for text in plt.gca().get_legend().get_texts():
        text.set_fontsize(24)

    plt.xticks(np.arange(0, np.max(np.asarray(data[xaxis])), 0.25))

    plt.title(title, fontsize=28)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.gcf().set_size_inches(9, 6)
    plt.yscale('symlog')
    g = plt.gca()


learning_starts = 5_000


def load_data(path, value="eval/mean_reward"):

    df = pd.read_csv(path)
    df = df[df["global_step"] > learning_starts]
    df = df[df[value] != None]
    # absolute all numeric values
    for col in df.columns:
        if col != "clip_method":
            df[col] = df[col].abs()
        if col == "global_step":
            df[col] = (df[col] / 1e5)

    return df


def plot(value="eval/mean_reward", title=None, xlabel=None, ylabel=None,
         xlim=None, ylim=None, **kwargs):
    data_list = []
    dict_counter = {'none': 0, 'soft': 0, 'hard': 0, 'soft_hard': 0, 'test': 0}
    files = os.listdir("export")
    cond_list = []
    for file_name in files:
        try:
            data = load_data("export/" + file_name, value=value)
            data_list.append(data)
            cond_list.append(file_name.split("-")[1])
            dict_counter[file_name.split("-")[1]] += 1
        except Exception as e:
            print("Error loading file: {}, error {}".format(file_name, e))
    plot_data(
        data_list,
        xaxis="global_step",
        value=value,
        condition="clip_method",
        errorbar=("ci", 95),
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        **kwargs
    )
    plt.savefig(f'plot_{title}.pdf', dpi=800)


if __name__ == "__main__":
    plot(value="train/bound_violations", title='Bound Violations in Training', xlabel=r'Environment Steps ($\times 100,000$)',
         ylabel='Bound Violations (in abs. val.)', xlim=(0.05, 1), ylim=(-1e-5, 5e3))
