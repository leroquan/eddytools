# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from utils_mitgcm import open_mitgcm_ds_from_config

import eddytools_core as et

def plot_map_eddies(snapshot_vel, snapshot_eddies, title, stream_density=6):
    x_plot, y_plot = np.meshgrid(snapshot_vel['lon'], snapshot_vel['lat'])
    u_plot = snapshot_vel['UVEL'].values
    v_plot = snapshot_vel['VVEL'].values

    plt.close('all')
    fig = plt.figure(figsize=(12, 7))

    # Plot background scalar field (e.g., UVEL)
    snapshot_vel["UVEL"].plot()

    # Plot streamlines
    plt.streamplot(x_plot, y_plot, u_plot, v_plot,
                   density=stream_density, color='black', linewidth=0.5,
                   arrowsize=0.7, arrowstyle='->')

    # Overlay eddies
    for i in range(len(snapshot_eddies)):

        if snapshot_eddies[i]['valid_eddy'] == -1:
            plt.scatter(snapshot_eddies[i]['lon'], snapshot_eddies[i]['lat'], c='red')
            plt.text(snapshot_eddies[i]['lon'], snapshot_eddies[i]['lat'], f'{round(snapshot_eddies[i]["winding"], 2)}')
            continue

        eddy_i = snapshot_eddies[i]['eddy_i']
        eddy_j = snapshot_eddies[i]['eddy_j']
        lon_eddy = snapshot_vel['lon'].values[eddy_i]
        lat_eddy = snapshot_vel['lat'].values[eddy_j]

        try:
            triang = tri.Triangulation(lon_eddy, lat_eddy)
            plt.tripcolor(triang, facecolors=np.full(len(triang.triangles), 1.0),
                          cmap=plt.cm.Greens, vmin=0, vmax=4, alpha=1)
        except Exception as e:
            plt.scatter(lon_eddy, lat_eddy, c='green')
        # Eddy center
        plt.scatter(snapshot_eddies[i]['lon'], snapshot_eddies[i]['lat'], c='black')

    plt.text(0.02, 0.98, f'Z={round(float(snapshot_vel.Depth.values), 2)}m', transform=plt.gca().transAxes, ha='left',
             va='top')
    plt.title(title)

    return fig


model = 'geneva_200m'

str_start_time = "2023-07-01T12:00:00.000000000"
str_end_time = "2023-07-01T22:00:00.000000000"

t_index = 3
outputpath = "./99-Outputs/nencioli/"

mitgcm_config, ds_to_plot = open_mitgcm_ds_from_config('../config.json', model)
grid_resolution_in_meter = ds_to_plot['XC'].values[1] - ds_to_plot['XC'].values[0]
data_aligned = et.detection_nencioli.preprocess_inputs(str_start_time, str_end_time, ds_to_plot, depth_index=0)

# Specify parameters for eddy detection
det_param = {
    "model": "MITgcm",
    "grid": "cartesian",
    "hemi": "north",
    "start_time": str_start_time,  # time range start
    "end_time": str_end_time,  # time range end
    "calendar": "standard",  # calendar, must be either 360_day or standard
    "lon1": data_aligned.lon.values.min(),  # minimum longitude of detection region
    "lon2": data_aligned.lon.values.max(),  # maximum longitude
    "lat1": data_aligned.lat.values.min(),  # minimum latitude
    "lat2": data_aligned.lat.values.max(),  # maximum latitude
    "res": grid_resolution_in_meter / 1000,  # resolution of the fields in km
    "min_dep": 1,  # minimum ocean depth where to look for eddies in m
    "no_long": False,  # If True, elongated shapes will not be considered
    "no_two": False,  # If True, eddies with two minima in the OW
    # parameter and a OW > OW_thr in between  will not
    # be considered
    "a": 3,  # u/v increase "a" points away from reversal
    "b": 3,  # find the velocity minimum within the searching area defined by
    # "b" around the points that satisfy the first two constraints
    "rad": 8,  # define the space window in which the algorithm looks for the eddy limits
}

test_eddies = et.detection_nencioli_winding.detect_UV_core(data_aligned,
                                                           det_param.copy(),
                                                           data_aligned["UVEL"].compute(),
                                                           data_aligned["VVEL"].compute(),
                                                           data_aligned["SPEED"].compute(),
                                                           t_index,
                                                           data_aligned['dxC'].values,
                                                           data_aligned['dyC'].values)

fig = plot_map_eddies(data_aligned.isel(time=t_index), test_eddies, '', stream_density=1)



