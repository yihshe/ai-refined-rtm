# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
# %%
# csv_path = {
#     "real": "/maps/ys611/ai-refined-rtm/data/real/BPWW_extract_2018_reshaped.csv",
#     "synthetic": "/maps/ys611/ai-refined-rtm/data/synthetic/20230529/synthetic.csv"
# }
csv_path = {
    "real": "/maps/ys611/ai-refined-rtm/data/real/BPWW_extract_2018_reshaped_train_scaled.csv",
    "synthetic": "/maps/ys611/ai-refined-rtm/data/synthetic/20230529/synthetic_train_scaled.csv"
}
SAVE_PATH = "/maps/ys611/ai-refined-rtm/visuals/3dplots/"
sets = ['real', 'synthetic']
dims = ['B04_RED', 'B08_NIR1', 'B12_SWI2']
dfs = {}
clouds = {}
for k in sets:
    dfs[k] = pd.read_csv(csv_path[k])
    clouds[k] = dfs[k][dims]
colors = {'real': 'red', 'synthetic': 'blue'}
# %%
# Create a new figure and add a 3D subplot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# Plot the point clouds with different colors
for k in sets:
    ax.scatter(clouds[k][dims[0]], clouds[k][dims[1]],
               clouds[k][dims[2]], c=colors[k], label=k)

# Set labels and legend
ax.set_xlabel(dims[0])
ax.set_ylabel(dims[1])
ax.set_zlabel(dims[2])
ax.legend()
plt.tight_layout()
# plt.savefig(os.path.join(
#     SAVE_PATH, '3d_real_v_syn_orig.png'), dpi=300)
plt.show()

# %%
traces = {}
for k in sets:
    traces[k] = go.Scatter3d(
        x=clouds[k][dims[0]],
        y=clouds[k][dims[1]],
        z=clouds[k][dims[2]],
        mode='markers',
        name=k,
        marker=dict(
            size=3,
            color=colors[k],  # set color of first point cloud to red
            opacity=0.1
        )
    )
# add axis title
layout = go.Layout(
    scene=dict(
        xaxis_title=dims[0],
        yaxis_title=dims[1],
        zaxis_title=dims[2]
    )
)

fig = go.Figure(data=[traces[k] for k in sets], layout=layout)

# Save the plot as an HTML file
# fig.write_html(os.path.join(SAVE_PATH, '3dplot_real_v_syn_orig.html'))
fig.write_html(os.path.join(SAVE_PATH, '3dplot_real_v_syn_scaled.html'))

# %%
