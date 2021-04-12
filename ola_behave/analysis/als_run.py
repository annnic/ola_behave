import glob
import os
import copy

import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askdirectory

from ola_behave.analysis.als_functions import neg_values, binner, remove_high_spd, smooth_speed
from ola_behave.io.tracks import load_track
from ola_behave.plotting.plots import filled_plot

# Allows a user to select file
root = Tk()
root.withdraw()
root.update()
path = askdirectory()
root.destroy()

# ##### parameters
fps = 30

# min_bins = 5
# change_times_s = [min_bins * 60, min_bins * 60 * 2, min_bins * 60 * 3, min_bins * 60 * 4]

epoque_lens = [5 * 60, 2 * 60, 2 * 60, 2 * 60, 2 * 60]
change_times_s = np.cumsum(epoque_lens)
change_conditions = ['E3-1', 'Odour-1', 'E3-2', 'Odour-2', 'E3-3']
# #####

roi = []
i = 0 # load each roi track
for csv_path in glob.glob(os.path.join(path, "*roi*.csv")):
    print("Loading {}".format(csv_path))
    displacement_internal, track_internal = load_track(csv_path)
    if i == 0:
        x = track_internal[:, 1]
        y = track_internal[:, 2]
        i = 1
        np.expand_dims(x, axis=0)
        np.expand_dims(y, axis=0)
    else:
        x = np.c_[x, track_internal[:, 1]]
        y = np.c_[y, track_internal[:, 2]]

    roi.append(int(csv_path[-5]))

# sort rois so each column is in order e.g. roi 0, 1, 2
x = x[:, np.argsort(roi)]
y = y[:, np.argsort(roi)]

# set up epoques and timings, tv = time vector
tv = np.arange(0, len(x[:, 1]))
tv_epoque = copy.copy(tv)

change_times_f = change_times_s*fps
j=0
for num, frame in enumerate(change_times_f):
    tv_epoque[j:frame] = num
    j = frame
tv_epoque[j:] = num+1
tv_s = tv / fps

# scatter plot of position coloured by epoque
fig, axs = plt.subplots(1, len(roi))
fig.suptitle('Position graph')
for roi_n in range(len(roi)):
    # axs[roi_n].scatter(x[:, roi_n], neg_values(y[:, roi_n]), c=tv[:], cmap="viridis", s=0.5)
    im = axs[roi_n].scatter(x[:, roi_n], neg_values(y[:, roi_n]), c=tv_epoque[:], cmap="Dark2", s=2.)
    axs[roi_n].set_xticks([])
    axs[roi_n].set_yticks([])
cbar = fig.colorbar(im, ticks=np.arange(0, len(change_conditions)))
cbar.ax.set_yticklabels(change_conditions)
plt.savefig(os.path.join(path, "{}_scatter_position.png".format(os.path.basename(path))))

# ## HEATMAP ##
fig, axs = plt.subplots(1, len(roi))
fig.suptitle('Denisty heatmap')
mappables = []
for roi_n in range(len(roi)):
    H, xedges, yedges = np.histogram2d(y[~np.isnan(y[:, roi_n]), roi_n], neg_values(x[~np.isnan(x[:, roi_n]), roi_n]),
                                       bins=[20, 3])
    mappables.append(H)

## the min and max values of all histograms
vmin = np.min(mappables)
vmax = np.max(mappables)

## second loop for visualisation
for ax, H in zip(axs.ravel(), mappables):
    im = ax.imshow(H, vmin=vmin, vmax=vmax, extent=[-0, 50, 0, 400])
    ax.axis("off")

## colorbar using solution from linked question
fig.colorbar(im, ax=axs.ravel())
plt.savefig(os.path.join(path, "{}_heatmap_position.png".format(os.path.basename(path))))


# halfway marked, plotting X position
x_halfway = np.empty(x.shape[1]) * np.nan
for row in range(x.shape[1]):
    x_halfway[row] = (np.nanmax(x[:, row]) - np.nanmin(x[:, row])) / 2

for row in range(x.shape[1]):
    fig.suptitle('Roi {}'.format(row))
    fig2, ax2 = filled_plot(tv_s, x[0:-1, row], change_times_s, x.shape[0]/fps)
    plt.plot([0, tv_s[-1]], [x_halfway[row], x_halfway[row]], color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("x position")
    plt.savefig(os.path.join(path,"{}_timeline_roi-{}.png".format(os.path.basename(path), row)))

# # Bin data in 5min bins
# smooth_win = fps * 60 * min_bins
# y_bin = np.zeros(x.shape)
# tv_bin, nothing = binner(tv, smooth_win, 1)
#
# fraction_on_top = np.zeros([len(tv_bin), 2])
#
# for row in range(y.shape[1]):
#     # y_bin[:, row] = (smooth_speed(y[:, row], win_size=smooth_win)).flatten()
#     y_bin_mean, y_bin_data = binner(y[:, row], smooth_win, 1)
#     on_top = (y_bin_data > x_halfway[row]) * 1
#     fraction_on_top[:, row] = np.mean(on_top, axis=1)
#     # fig1, ax1 = plt.subplots()
#     # plt.xlabel("Time (s)")
#     # plt.ylabel("Fraction time on top side")
#     # plt.title("Fraction time on top side roi {}".format(row))
#     # ax1.bar(tv_bin, fraction_on_top, width=smooth_win/fps, bottom=None, align='center')
#     # plt.ylim([0, 1])
#
#     fig1, ax1 = plt.subplots()
#     plt.xlabel("Time (s)")
#     plt.ylabel("Fraction time on each side")
#     plt.title("Fraction time on each side roi {}".format(row))
#     ax1.bar(tv_bin, (fraction_on_top[:, row]), width=smooth_win / fps, bottom=None, align='center',
#             color='cornflowerblue')
#     ax1.bar(tv_bin, (1 - fraction_on_top[:, row]) * -1, width=smooth_win / fps, bottom=None, align='center',
#             color='cornflowerblue')
#     plt.plot([0, tv[-1]], [0, 0], color='grey')
#     plt.ylim([-1, 1])
#     plt.savefig(os.path.join(path,
#                              "{}_fraction_time_each_side_{}min_bins_roi-{}.png".format(os.path.basename(path), min_bins,
#                                                                                        row)))
#
# fraction_on_top_df = pd.DataFrame(fraction_on_top, columns=['roi_0', 'roi_1'], index=pd.Index(tv_bin))
# fraction_on_top_df.to_csv(os.path.join(path, "{}_als.csv".format(os.path.basename(path))))

print("finished")
