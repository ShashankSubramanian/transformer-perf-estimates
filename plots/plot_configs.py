import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import numpy as np

def plot_configs(n_values, time_total, time_parts_normalized, 
                 mem_values, m_values, tp_values, dp_values, pp_values, 
                 tp1_values=[], tp2_values=[],
                 use_log_scale=True):
    '''
    Inputs: 
    n_values: list of #GPUs
    time_total: list of total training time 
    time_parts_normalized: list of time components in fractions
    mem_values: list of memory 
    m_values: list of microbatch sizes
    tp_values: list of TP 
    dp_values: list of DP
    pp_values: list of PP
    tp1_values: list of TP1 in 2D parallelism
    tp2_values: list of TP2 in 2D parallelism
    use_log_scale: bool that tells the plot that #GPUs are on a log-scale

    Returns:
    Plot of optimal parallel configs vs #GPUs and time vs #GPUs broken down into its 
    individual components
    '''
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    bar_width = 0.2
    if use_log_scale: #
        log_n_values = np.log2(n_values)
    else:
        log_n_values = (n_values)

    # First plot: m, tp, dp, pp as a function of n
    if len(tp1_values) == 0: # 1D
        plt.bar(log_n_values - 1.5 * bar_width, m_values, bar_width, label='#Microbatches', color='dodgerblue')
        plt.bar(log_n_values - 0.5 * bar_width, tp_values, bar_width, label='TP', color='salmon')
        plt.bar(log_n_values + 0.5 * bar_width, dp_values, bar_width, label='DP', color='limegreen')
        plt.bar(log_n_values + 1.5 * bar_width, pp_values, bar_width, label='PP', color='orange')
    else: # need to plot additional TP
        plt.bar(log_n_values - 1.5 * bar_width, m_values, bar_width, label='#Microbatches', color='dodgerblue')
        plt.bar(log_n_values - 0.5 * bar_width, tp1_values, bar_width, label=r'$n_1$ TP', color='salmon')
        plt.bar(log_n_values + 0.5 * bar_width, tp2_values, bar_width, label=r'$n_2$ TP', color='mediumorchid')  # Added next to TP
        plt.bar(log_n_values + 1.5 * bar_width, dp_values, bar_width, label=r'$n_d$ DP', color='limegreen')
        plt.bar(log_n_values + 2.5 * bar_width, pp_values, bar_width, label=r'$n_p$ PP', color='orange')

    plt.xlabel('#GPUs', fontsize=16)
    plt.ylabel('Values', fontsize=16)
    plt.title('Parallelization Configuration vs #GPUs', fontsize=16)
    plt.xticks(log_n_values, n_values, fontsize=14)
    plt.yscale('log', base=2)
    top = max(max(m_values), max(tp_values), max(dp_values), max(pp_values))
    top2 = int(np.ceil(np.log2(top)))+1
    plt.ylim([0.1, top * 2])
    plt.yticks([2**i for i in range(0, top2)], [2**i for i in range(0, top2)], fontsize=14)
    if use_log_scale:
        plt.xlim([np.log2(n_values[0]) - 1, np.log2(n_values[-1]) + 1])
    plt.legend(fontsize=14, loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()

    # add mem values as a twin
    ax3 = plt.gca().twinx()
    ax3.plot(log_n_values, mem_values, color='black', linewidth=1, linestyle='--', marker='o', label='Memory Usage')
    ax3.set_ylabel('Memory Usage (in GB)', fontsize=16, color='black')
    ax3.tick_params(axis='y', labelcolor='black', labelsize=14)
    ax3.set_ylim(0, 200)

    lines, labels = ax3.get_legend_handles_labels()
    plt.legend(lines, labels, fontsize=12, loc='upper right')
    plt.tight_layout()

    # Second plot: Time vs n as a stacked bar plot with percentages and total time line
    ax1 = plt.subplot(2, 1, 2)
    bar_width = 0.5

    colors = ['dodgerblue', 'salmon', 'limegreen', 'orange', 'slateblue', 'skyblue']
    labels = ['Compute', 'TP Comm', 'PP Bubble', 'DP Comm', 'Memory', 'PP Comm']
    bottom = np.zeros(len(n_values))

    for i, (height, color, label) in enumerate(zip(time_parts_normalized, colors, labels)):
        p = ax1.bar(log_n_values, height, bottom=bottom, label=label, color=color, width=bar_width)
        bottom += height

        # Add percentage text
        for j, h in enumerate(height):
            if h > 1:  # Only show percentage if it's greater than 1%
                ax1.text(log_n_values[j], bottom[j] - h/2, f'{h:.0f}%', ha='center', va='center', color='white', fontsize=9)

    ax1.set_xlabel('#GPUs', fontsize=16)
    ax1.set_ylabel('Normalized Time (%)', fontsize=16)
    ax1.set_title('Time vs #GPUs', fontsize=16)
    ax1.set_xticks(log_n_values)
    ax1.set_xticklabels(n_values, fontsize=14)
    if use_log_scale:
        ax1.set_xlim([np.log2(n_values[0]) - 1, np.log2(n_values[-1]) + 1])
    ax1.legend(fontsize=12, loc='upper left')
    ax1.tick_params(axis='y', labelsize=14)

    ax1.set_ylim(0, 105)  # Set to slightly over 100% to give some headroom

    # add total time as well
    ax2 = ax1.twinx()
    ax2.plot(log_n_values, time_total, color='black', linewidth=1, linestyle='--', marker='o', label='Total Time')
    ax2.set_ylabel('Time per Iteration (s)', fontsize=16, color='black')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=14)
    ax2.set_yscale('log', base=2)
    ax2.set_ylim(min(time_total) / 2, max(time_total) * 2)
    lines, labels = ax2.get_legend_handles_labels()
    ax1.legend(fontsize=12, loc='upper left')
    ax2.legend(lines, labels, fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.show()
