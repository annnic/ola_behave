import math

import matplotlib.pyplot as plt


def filled_plot(tv_internal, speed, change_times_unit, day_unit):
    days_to_plot = math.ceil(tv_internal[-1]/day_unit)
    figa, ax = plt.subplots(figsize=(15, 5))
    ax.plot(tv_internal[0:-1], speed[:], color='black')
    for day_n in range(days_to_plot):
        ax.axvspan(0+day_unit*day_n, change_times_unit[0]+day_unit*day_n, color='lightblue', alpha=0.5, linewidth=0)
        ax.axvspan(change_times_unit[0]+day_unit*day_n, change_times_unit[1]+day_unit*day_n, color='wheat', alpha=0.5,
                   linewidth=0)
        ax.axvspan(change_times_unit[2]+day_unit*day_n, change_times_unit[3]+day_unit*day_n, color='wheat', alpha=0.5,
                   linewidth=0)
        ax.axvspan(change_times_unit[3]+day_unit*day_n, day_unit + day_unit * day_n, color='lightblue', alpha=0.5,
                   linewidth=0)
    ax.set_xlim([8, days_to_plot*day_unit])
    return figa, ax
