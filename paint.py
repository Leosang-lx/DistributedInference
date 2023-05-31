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
    fontsize = 20
    plt.figure(figsize=(16, 8))
    for idx, sizes in enumerate(sizess):
        xs = list(range(len(sizes)))
        ys = np.asarray(sizes) / 1024  # unit: KB
        plt.plot(xs, ys, label=labels[idx], linewidth=2.5)
    plt.xlabel('Layers', fontsize=24)
    plt.ylabel('Transmission size (KB)', fontsize=24)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.tight_layout()
    # plt.title('Transmission size of layers')
    if file_name is not None:
        plt.savefig('D:/华为云盘/毕设/final/figures/' + file_name + '.pdf', bbox_inches='tight')
    plt.show()




def show_workers_transmission_size(sizes: list, file_name=None):
    n_device = len(sizes)
    fontsize = 20
    plt.figure(figsize=(10, 5))
    for i in range(n_device):
        plt.subplot(n_device, 1, i + 1)
        xs = list(range(len(sizes[i])))
        ys = np.asarray(sizes[i])
        ys = ys / 1024  # unit: KB
        plt.plot(xs, ys)
        plt.title(f'worker{i+1}')
    # plt.xticks(fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
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


def show_inference_latency(file_name=None):
    fontsize = 12
    labels = ['Local', 'BOD', 'MPBD', 'PCC', 'PPC']
    worker2 = [3.348966360092163, 3.3469395637512207, 2.6024765968322754, 2.0084524154663086]
    worker3 = [3.466867208480835, 2.2908759117126465, 1.9593393802642822, 1.5427675247192383]

    # worker2 = [37.49709892272949, 37.57251453399658, 20.96573042869568, 16.335726737976074]
    # worker3 = [38.8679883480072, 17.84951162338257, 14.576863765716553, 10.384487390518188]
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    plt.figure(figsize=(8, 2.5))
    plt.grid(axis='y', ls='--', zorder=0)
    plt.bar(0, 3.59509, width, label='single', color='#A0A0A0', linewidth=1.0, edgecolor='black', zorder=10)
    # plt.bar(0, 33.34104, width, label='single', color='#A0A0A0', linewidth=1.0, edgecolor='black', zorder=10)
    plt.bar(x[1:] - width / 2, worker2, width, label='2 workers', color='#FE817D', linewidth=1.0, edgecolor='black', zorder=10)
    plt.bar(x[1:] + width / 2, worker3, width, label='3 workers', color='#81B8DF', linewidth=1.0, edgecolor='black', zorder=10)


    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Latency (s)', fontsize=fontsize)
    plt.xticks(x, labels, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=10)
    if file_name is not None:
        plt.savefig('D:/华为云盘/毕设/final/figures/' + file_name + '.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # show_inference_latency('latency_googlenet_224')
    fontsize = 12
    labels = ['Local', '2 workers', '3 workers']
    latencies = [50.44, 23.25814723968506, 15.812919855117798]
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    plt.figure(figsize=(6, 3))
    plt.grid(axis='y', ls='--', zorder=0)
    plt.bar(x, latencies, width, color='#71A8CF', linewidth=1.0, edgecolor='black', zorder=10)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Latency (s)', fontsize=fontsize)
    # plt.title('Inference latency of GoogLeNet with input shape of (1,3,224,224)', fontsize=fontsize)
    # plt.title('Distributed inference latency of VGG16', fontsize=fontsize)
    plt.xticks(x, labels, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.legend(fontsize=fontsize)
    file_name = 'latency_vgg16_224'
    # file_name = None
    if file_name is not None:
        plt.savefig('D:/华为云盘/毕设/final/figures/' + file_name + '.pdf', bbox_inches='tight')
    plt.show()