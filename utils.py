import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

def show_grid(grid):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = grid.shape
    width, height = 1.0/ncols, 1.0/nrows

    for (i, j), val in np.ndenumerate(grid):
        color = 'grey' if (i, j) == (0, 0) or (i, j) == (3, 3) else 'white'
        tb.add_cell(i, j, width, height, text="{0:.2f}".format(val),
                    loc='center', facecolor=color)

    ax.add_table(tb)
    return ax
