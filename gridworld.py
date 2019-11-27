import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from IPython import embed

DISCOUNT = 1
GRID_SIZE = 4
GRID = np.arange(GRID_SIZE*GRID_SIZE).reshape(GRID_SIZE, GRID_SIZE)
ACTIONS = [
            np.array([0, -1]), #left
            np.array([0,  1]), #right
            np.array([-1, 0]), #up
            np.array([1,  0])  #down
]
ACTION_PROB = 0.25
           
def step(state, action):
    if state == [0, 0]:
        return [0, 0], 0.0
    if state == [3, 3]:
        return [3, 3], 0.0

    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = -1.0
    return next_state, reward

def show_grid(grid):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = grid.shape
    width, height = 1.0/ncols, 1.0/nrows

    for (i, j), val in np.ndenumerate(grid):
        color = 'grey' if (i, j) == (0, 0) or (i, j) == (3, 3) else 'white'
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor=color)

    ax.add_table(tb)
    plt.show()

value = np.zeros_like(GRID, dtype=np.float64)

for _ in range(10):
    new_value = np.zeros_like(GRID, dtype=np.float64)
    for (i, j) in np.ndindex(GRID.shape):
        for action in ACTIONS:
            (next_i, next_j), reward = step([i, j], action)
            new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j])
    value = new_value
    show_grid(value)
