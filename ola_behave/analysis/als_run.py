import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askdirectory
import pandas as pd

from ola_behave.analysis.als_functions import neg_values, binner, remove_high_spd, smooth_speed
from ola_behave.io.tracks import load_track
# from ola_behave.utils.timings import infer_tv
from ola_behave.plotting.plots import filled_plot

# Allows a user to select file
root = Tk()
root.withdraw()
root.update()
path = askdirectory()
root.destroy()

# parameters
fps = 30

roi = []
i = 0
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

x = x[:, np.argsort(roi)]
y = y[:, np.argsort(roi)]

tv = np.arange(0, len(x[:, 1]))
tv = tv / fps

fig, axs = plt.subplots(1, len(roi))
fig.suptitle('Position graph')
for roi_n in range(len(roi)):
    axs[roi_n].scatter(x[:, roi_n], neg_values(y[:, roi_n]), c=tv[:], cmap="viridis", s=0.5)
# fig.colorbar(mpl.cm.ScalarMappable(cmap="viridis"), cax=axs[roi_n+1], orientation='vertical', label='% of  time')
plt.savefig(os.path.join(path, "{}_scatter_position.png".format(os.path.basename(path))))

fig, axs = plt.subplots(1, len(roi))
fig.suptitle('Denisty heatmap')
mappables = []
for roi_n in range(len(roi)):
    H, xedges, yedges = np.histogram2d(y[~np.isnan(y[:, roi_n]), roi_n], neg_values(x[~np.isnan(x[:, roi_n]), roi_n]),
                                       bins=[20, 3])
    # ax.set_xlabel("roi {}".format(roi_n))
    # ax.set_aspect("equal")
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

min_bins = 5
change_times_s = [min_bins * 60, min_bins * 60 * 2, min_bins * 60 * 3, min_bins * 60 * 4]

y_halfway = [np.nan, np.nan]
for row in range(x.shape[1]):
    y_halfway[row] = (np.nanmax(y[:, row]) - np.nanmin(y[:, row])) / 2

for row in range(x.shape[1]):
    fig2, ax2 = filled_plot(tv, y[0:-1, row], change_times_s, x.shape[0] / fps)
    # fig2, ax2 = filled_plot(tv, np.diff(y[:, row]), change_times_s, x.shape[0]/fps)
    plt.plot([0, tv[-1]], [y_halfway[row], y_halfway[row]], color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("y position")
    plt.savefig(os.path.join(path,
                             "{}_timeline_{}min_bins_roi-{}.png".format(os.path.basename(path), min_bins, row)))

# Bin data in 5min bins
smooth_win = fps * 60 * min_bins
y_bin = np.zeros(x.shape)
tv_bin, nothing = binner(tv, smooth_win, 1)

fraction_on_top = np.zeros([len(tv_bin), 2])

for row in range(y.shape[1]):
    # y_bin[:, row] = (smooth_speed(y[:, row], win_size=smooth_win)).flatten()
    y_bin_mean, y_bin_data = binner(y[:, row], smooth_win, 1)
    on_top = (y_bin_data > y_halfway[row]) * 1
    fraction_on_top[:, row] = np.mean(on_top, axis=1)
    # fig1, ax1 = plt.subplots()
    # plt.xlabel("Time (s)")
    # plt.ylabel("Fraction time on top side")
    # plt.title("Fraction time on top side roi {}".format(row))
    # ax1.bar(tv_bin, fraction_on_top, width=smooth_win/fps, bottom=None, align='center')
    # plt.ylim([0, 1])

    fig1, ax1 = plt.subplots()
    plt.xlabel("Time (s)")
    plt.ylabel("Fraction time on each side")
    plt.title("Fraction time on each side roi {}".format(row))
    ax1.bar(tv_bin, (fraction_on_top[:, row]), width=smooth_win / fps, bottom=None, align='center',
            color='cornflowerblue')
    ax1.bar(tv_bin, (1 - fraction_on_top[:, row]) * -1, width=smooth_win / fps, bottom=None, align='center',
            color='cornflowerblue')
    plt.plot([0, tv[-1]], [0, 0], color='grey')
    plt.ylim([-1, 1])
    plt.savefig(os.path.join(path,
                             "{}_fraction_time_each_side_{}min_bins_roi-{}.png".format(os.path.basename(path), min_bins,
                                                                                       row)))

fraction_on_top_df = pd.DataFrame(fraction_on_top, columns=['roi_0', 'roi_1'], index=pd.Index(tv_bin))
fraction_on_top_df.to_csv(os.path.join(path, "{}_als.csv".format(os.path.basename(path))))

print("finished")
