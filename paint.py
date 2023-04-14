from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt


def show_time_intervals(data):
    """
    Parameters
    ----------
    data : dict
        a list of timepoints records the start time and end time of each subtask of all workers
    """
    labels = ['worker{}'.format(i+1) for i in range(len(data))]
    for i in range(len(data)):  # cal diff of the time points
        data[i] = np.diff(data[i])

    y_max = max([d.sum() for d in data])
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.invert_yaxis()
    ax.xaxis.set_visible(True)
    ax.set_xlim(0, y_max)

    color = ['#40E0D0', '#A9A9A9']
    colname = ['computing', 'waiting']
    max_len = max(len(i) for i in data)
    starts = np.asarray([0.0 for _ in range(len(labels))])

    for i in range(max_len):
        c = color[i%2]
        label = colname[i%2]
        values = []
        for worker in data:
            if i < len(worker):
                values.append(worker[i])
            else:
                values.append(0)
        values = np.asarray(values)
        ax.barh(labels, values, left=starts, height=0.6, label=label, color=c)
        starts += values
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    # omit redundant legend
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1), loc='lower left')
    plt.show()


def show_transmission_size(sizes: list):
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
    plt.show()
    for i in range(n_device):
        xs = list(range(len(sizes[i])))
        ys = np.asarray(sizes[i])
        ys = ys / 1024  # unit: KB
        plt.plot(xs, ys, label=f'worker{i+1}')
    plt.legend()
    plt.suptitle('Transmission size')
    plt.show()