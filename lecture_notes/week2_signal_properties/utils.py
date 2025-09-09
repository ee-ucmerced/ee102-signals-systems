import numpy as np
import matplotlib.pyplot as plt
def plot_signal_transformations(time_data, plots):
    """
    Custom plotting code for notes
    """
    fig, axs = plt.subplots(2, 4, figsize=(30, 15), sharex=True, sharey=True)
    for i, j, y, title in plots:
        for spine in axs[i, j].spines.values():
            spine.set_visible(False)
        axs[i, j].plot(time_data, y, lw=2, c="k")
        axs[i, j].set_title(title)
        axs[i, j].set(xlabel='Time (t)', ylabel='$\\bar{x}(t)$')
        axs[i, j].grid()
        axs[i, j].tick_params(axis='both', which='major', labelsize=22)
        axs[i, j].title.set_fontsize(24)
        axs[i, j].xaxis.label.set_fontsize(24)
        axs[i, j].yaxis.label.set_fontsize(28)
        # Draw vertical 0-axis
        axs[i, j].axvline(0, color='k', lw=4)
        # Draw horizontal 0-axis
        axs[i, j].axhline(0, color='k', lw=4)
        # Set all other spines to lw=2 except left and bottom
        for spine in ['top', 'right']:
            axs[i, j].spines[spine].set_linewidth(2)
        axs[i, j].spines['left'].set_linewidth(3)
        axs[i, j].spines['bottom'].set_linewidth(3)
        axs[i, j].xaxis.set_tick_params(which='both',
                                        labelbottom=True,
                                        )
        axs[i, j].yaxis.set_visible(True)
    y_min, y_max = axs[0,0].get_ylim()
    for ax in axs.flat:
        ax.set_ylim(y_min, y_max)
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("figs/signal_transformations.png")
    plt.show()
