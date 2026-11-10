

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


def plot_spatial_scatterpie_highlight(file_path, pie_radius=0.45):
    df = pd.read_csv(file_path)

    x_coords = df.iloc[:, 1]
    y_coords = df.iloc[:, 2]
    cell_types = list(df.columns[3:])

    highlight_cells = {
        'Cancer.clone.A': '#E31A1C',
        'Cancer.clone.B': '#FF7F00'
    }

    other_colors = ['#D3D3D3', '#E0E0E0', '#DCDCDC', '#F5F5F5', '#C0C0C0', '#BDBDBD']

    colors = []
    other_idx = 0
    for ct in cell_types:
        if ct in highlight_cells:
            colors.append(highlight_cells[ct])
        else:
            colors.append(other_colors[other_idx % len(other_colors)])
            other_idx += 1

    color_dict = dict(zip(cell_types, colors))

    fig, ax = plt.subplots(figsize=(12, 10))

    for i in range(len(df)):
        x = x_coords[i]
        y = y_coords[i]
        sizes = df.iloc[i, 3:].values.astype(float)

        if sizes.sum() == 0: continue

        start_angle = 90
        for j, size in enumerate(sizes):
            if size > 0:
                extent = size * 360
                w = Wedge(center=(x, y), r=pie_radius,
                          theta1=start_angle, theta2=start_angle + extent,
                          facecolor=colors[j],
                          edgecolor='white', linewidth=0.1)
                ax.add_patch(w)
                start_angle += extent

    ax.set_xlim(x_coords.min() - 1, x_coords.max() + 1)
    ax.set_ylim(y_coords.min() - 1, y_coords.max() + 1)
    ax.set_aspect('equal')

    legend_elements = []
    for name, color in highlight_cells.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          label=f"★ {name}", markerfacecolor=color, markersize=12))
    for ct in cell_types:
        if ct not in highlight_cells:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              label=ct, markerfacecolor=color_dict[ct], markersize=8))

    ax.legend(handles=legend_elements, title="Cell Types",
              loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

    plt.title('Spatial Proportion: Cancer Clones Highlighted', fontsize=15)
    plt.tight_layout()
    plt.show()


csv_file = 'PDAC_A_ST1_SPOTlight_W_omega_with_xy.csv'
plot_spatial_scatterpie_highlight(csv_file)