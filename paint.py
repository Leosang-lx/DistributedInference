from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt


def show_time_intervals(start, end, data, file_name=None):
    """
    Parameters
    ----------
    data : dict
        a list of timepoints records the start time and end time of each subtask of all workers
    """
    labels = ['worker{}'.format(i+1) for i in range(len(data))]
    begins = [d[0] for d in data]
    ends = [d[-1] for d in data]
    for i in range(len(data)):  # cal diff of the time points
        data[i] = np.diff(data[i])

    y_max = end - start
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.invert_yaxis()
    ax.xaxis.set_visible(True)
    ax.set_xlim(0, y_max)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('time (s)', fontsize=12)

    color = ['#40E0D0', '#A9A9A9']
    colname = ['computing', 'waiting']
    max_len = max(len(i) for i in data)
    starts = np.asarray([0.0 for _ in range(len(labels))])

    values = np.asarray(begins) - start
    ax.barh(labels, values, left=starts, height=0.5, label=colname[1], color=color[1])
    starts += values
    for i in range(max_len):
        c = color[i % 2]
        label = colname[i % 2]
        values = []
        for worker in data:
            if i < len(worker):
                values.append(worker[i])
            else:
                values.append(0)
        values = np.asarray(values)
        ax.barh(labels, values, left=starts, height=0.5, label=label, color=c)
        starts += values
    values = np.asarray([end for _ in range(len(labels))]) - np.asarray(ends)
    ax.barh(labels, values, left=starts, height=0.5, label=colname[1], color=color[1])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    # omit redundant legend
    plt.rcParams.update({'font.size':12})
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1), loc='lower left')
    plt.tight_layout()
    if file_name is not None:
        plt.savefig('D:/华为云盘/毕设/final/figures/' + file_name + '.pdf', bbox_inches='tight')
    plt.show()


def show_transmission_size(sizess: list, labels: list, file_name=None):
    fontsize = 16
    plt.figure(figsize=(10, 5))
    for idx, sizes in enumerate(sizess):
        xs = list(range(len(sizes)))
        ys = np.asarray(sizes) / 1024  # unit: KB
        plt.plot(xs, ys, label=labels[idx])
    plt.xlabel('Transmission size (KB)', fontsize=fontsize)
    plt.ylabel('Layers', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    # plt.title('Transmission size of layers')
    if file_name is not None:
        plt.savefig('D:/华为云盘/毕设/final/figures/' + file_name + '.pdf', bbox_inches='tight')
    plt.show()




def show_workers_transmission_size(sizes: list, file_name=None):
    n_device = len(sizes)
    plt.figure(figsize=(12, 6))
    for i in range(n_device):
        plt.subplot(n_device, 1, i + 1)
        xs = list(range(len(sizes[i])))
        ys = np.asarray(sizes[i])
        ys = ys / 1024  # unit: KB
        plt.plot(xs, ys)
        plt.title(f'worker{i+1}')
    plt.suptitle('Transmission size')
    plt.tight_layout()
    if file_name is not None:
        plt.savefig('D:/华为云盘/毕设/final/figures/' + file_name + '.pdf', bbox_inches='tight')
    plt.show()
    for i in range(n_device):
        xs = list(range(len(sizes[i])))
        ys = np.asarray(sizes[i])
        ys = ys / 1024  # unit: KB
        plt.plot(xs, ys, label=f'worker{i+1}')
    plt.legend()
    plt.suptitle('Transmission size')
    plt.show()