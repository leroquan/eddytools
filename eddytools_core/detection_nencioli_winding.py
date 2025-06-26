'''detection
Collection of functions needed for the detection of mesoscale eddies
based on the Nencioli (2010) method. The data is assumed to have been
interpolated with the `interp.py` module of this package or at least
needs to have the same structure.
'''

import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
import xarray as xr
import cftime as cft
import itertools
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
try:
    import multiprocessing as mp
except:
    print("multiprocessing not possible")
try:
    from dask import bag as dask_bag
except:
    print("Working without dask bags.")


def define_detection_parameter(str_start_time, str_end_time, data_aligned, grid_resolution, a, b, rad):
    return {
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
            "res": grid_resolution,  # resolution of the fields in km
            "min_dep": 1,  # minimum ocean depth where to look for eddies in m
            "no_long": False,  # If True, elongated shapes will not be considered
            "no_two": False,  # If True, eddies with two minima in the OW
            # parameter and a OW > OW_thr in between  will not
            # be considered
            "a": a, # u/v increase "a" points away from reversal
            "b": b, # find the velocity minimum within the searching area defined by
                    # "b" around the points that satisfy the first two constraints
            "rad": rad, # define the window in which eddy it looks for the eddy limits
        }

def select_date_and_depth(data, str_start_time: str, str_end_time:str, selected_depth_index: int = 0):
    start_date_analysis = np.datetime64(str_start_time)
    end_date_analysis = np.datetime64(str_end_time)
    
    data_cropped = data.sel(time=slice(start_date_analysis, end_date_analysis))
    data_cropped = data_cropped.isel(Z=selected_depth_index)

    return data_cropped

    
def format_data(data):
    temp_ini = data.THETA.isel(time=0)
    mask = temp_ini.where(abs(temp_ini) > 1e-10).values

    VVEL_new = (
    data["VVEL"]
        .rename({"YG": "lat", "XC": "lon", "Z": "Depth"})
        .assign_coords(lat=data["YC"].values)
    )
    UVEL_new = (
        data["UVEL"]
        .rename({"XG": "lon", "YC": "lat", "Z": "Depth"})
        .assign_coords(lon=data["XC"].values)
    )
    
    data_aligned = xr.Dataset(
        {
            "UVEL": mask*UVEL_new,
            "VVEL": mask*VVEL_new,
        },
        coords={
            "lon": data["XC"].values,
            "lat": data["YC"].values,
            "Depth": data["Z"].values,
        },
    )
    
    data_aligned["SPEED"] = np.sqrt(data_aligned["UVEL"] ** 2 + data_aligned["VVEL"] ** 2)

    return data_aligned


def preprocess_inputs(str_start_time, str_end_time, data, depth_index):
    data_cropped = select_date_and_depth(data,
                                str_start_time, 
                                str_end_time, 
                                depth_index)

    return format_data(data_cropped)


def check_if_in_polygon(x_center: float, y_center: float, x_isoline, y_isoline):
    try:
        p = np.array([x_isoline, y_isoline]).T
        hullp = ConvexHull(p)
        poly_path = Path(p[hullp.vertices])
        is_in_polygon = poly_path.contains_point((x_center, y_center))
    except Exception as e:
        is_in_polygon = False

    return is_in_polygon


def inpolygon2D(points_i, points_j, x, y):
    try:
        p = np.array([x, y]).T
        hullp = ConvexHull(p)
        poly_path = Path(p[hullp.vertices, :])
        is_in_polygon = poly_path.contains_points(np.array([points_i, points_j]).T)
    except Exception as e:
        is_in_polygon = False

    return is_in_polygon

def normalize_angle(angle):
    """Normalize angle to [-180, 180] degrees."""
    angle = (angle + 180) % 360 - 180
    return angle


def check_winding_angle(isoline_x, isoline_y, winding_thres=360, baddir_thres=90, d_thres=500):
    """
    Detects eddies in a streamline by analyzing its winding angle.

    Parameters:
        isoline_x (array-like): X coordinates of the isoline.
        isoline_y (array-like): Y coordinates of the isoline.
        winding_thres (float): Threshold winding angle to declare an eddy (degrees).
        baddir_thres (float): Threshold for angle deviation in wrong direction (degrees).
        d_thres (float): Distance threshold for closed loop detection.

    Returns:
        (bool, float, [], []): (eddy, winding, contour_x, contour_y)
    """
    winding = 0.0
    baddir = 0.0
    dir_sign = 0
    eddy = False
    contour_x = []
    contour_y = []
    min_dist = float('inf')

    # Initial direction
    dx0 = isoline_x[1] - isoline_x[0]
    dy0 = isoline_y[1] - isoline_y[0]
    prev_angle = np.degrees(np.arctan2(dy0, dx0))

    for i in range(len(isoline_x) - 1, 1, -1):
        angle_diff, prev_angle, winding  = increment_winding_angle(i, prev_angle, isoline_x, isoline_y, winding)

        if not check_direction_consistency(angle_diff, baddir, baddir_thres, dir_sign):
            break

        # Eddy detection logic
        if abs(winding) > winding_thres:
            dist_to_start = np.hypot(isoline_x[i] - isoline_x[-1], isoline_y[i] - isoline_y[-1])
            if eddy and dist_to_start > min_dist:
                contour_x = isoline_x[i+1:]
                contour_y = isoline_y[i+1:]
                break
            if dist_to_start < d_thres:
                eddy = True
                contour_x = isoline_x[i:]
                contour_y = isoline_y[i:]
                min_dist = dist_to_start

    return eddy, winding, contour_x, contour_y


def check_direction_consistency(angle_diff, baddir, baddir_thres, dir_sign):
    """
    Checks if the flow diverges too much from a perfect circle.
    """
    is_direction_consistent = True
    new_dir = np.sign(angle_diff)
    if dir_sign == 0 and new_dir != 0:
        dir_sign = new_dir
    elif new_dir != 0 and new_dir != dir_sign:
        baddir += angle_diff
        if abs(baddir) > baddir_thres:
            is_direction_consistent = False
    else:
        baddir = 0

    return is_direction_consistent


def increment_winding_angle(i, prev_angle, stline_x, stline_y, winding):
    dx = stline_x[i - 1] - stline_x[i]
    dy = stline_y[i - 1] - stline_y[i]
    curr_angle = np.degrees(np.arctan2(dy, dx))
    angle_diff = normalize_angle(curr_angle - prev_angle)
    prev_angle = curr_angle
    winding += angle_diff

    return angle_diff, prev_angle, winding


def get_eddy_contour(u, v, lon, lat, ec_lon, ec_lat, d_thres):
    """
    Get the largest contour around eddy center fitting these criterions:
    1) detected eddy center inside the polygon
    2) closed contour
    3) valid winding angle

    Returns
    -------
    eddy_i, eddy_j, eddy_mask, winding, isolines
    """
    isolines, isolines_max = generate_streamlines(u, v, lat, lon, level_density=6)


    # sort the contours accroding to their maximum latitude; this way the first
    # closed contour across which velocity increases will also be the largest
    # one (it's the one which extend further north).
    sorted_iso = np.argsort(isolines_max)[::-1]

    # Conditions to have the largest closed contour around the center
    # (isolines already sorted by maximum latitude)
    # 1) detected eddy center inside the polygon
    # 2) closed contour
    # 3) valid winding angle (Chaigneau 2008)
    eddy_lim = []
    for i in range(len(isolines)-1 ):
        ii = sorted_iso[i]
        xdata = isolines[ii][:,0] # vertex lon's ,
        ydata = isolines[ii][:,1] # vertex lat's
        winding=-1

        # 1) detected eddy center inside the polygon
        try:
            is_in_polygon = check_if_in_polygon(ec_lon, ec_lat, xdata, ydata)
        except Exception as e:
            is_in_polygon = False

        if not is_in_polygon:
            continue

        # 2) closed contour
        #if not ((xdata[0] == xdata[-1]) & (ydata[0] == ydata[-1])):
        #    continue

        # 3) valid winding angle
        winding = 0
        valid_winding, winding, contour_x, contour_y = check_winding_angle(xdata, ydata, winding_thres=360,
                                                                           baddir_thres=90, d_thres=d_thres)
        if valid_winding:
            # Found the eddy contour
            eddy_lim = [contour_x, contour_y]
            break

    eddy_i, eddy_j, eddy_mask =  None, None, None
    if eddy_lim:
        mask = inpolygon2D(lon.flatten(), lat.flatten(), eddy_lim[0], eddy_lim[1])
        eddy_mask = mask.reshape(np.shape(lon))
        eddy_j = np.where(eddy_mask)[0]
        eddy_i = np.where(eddy_mask)[1]

    return eddy_i, eddy_j, eddy_mask, winding, isolines


def build_streamlines(segments, tol=1e0):
    # Each segment is a (2, 2) array: [[x1, y1], [x2, y2]]
    start_points = np.array([seg[0] for seg in segments])
    used = np.zeros(len(segments), dtype=bool)

    # Build KDTree for fast spatial search
    start_points_tree = cKDTree(start_points)

    streamlines = []

    for i in range(len(segments)):
        if used[i]:
            continue
        streamline = [segments[i][0], segments[i][1]]
        used[i] = True
        end = segments[i][1]

        # Follow the chain forward
        while True:
            dists, idxs = start_points_tree.query(end, k=5, distance_upper_bound=tol)
            found = False
            for dist, idx in zip(dists, idxs):
                if idx == len(start_points) or used[idx]:
                    continue
                if np.linalg.norm(start_points[idx] - end) < tol:
                    streamline.append(segments[idx][1])
                    used[idx] = True
                    found = True
                    end = segments[idx][1]
                    break
            if not found:
                break

        streamlines.append(np.array(streamline))

    return streamlines


def generate_streamlines( u, v, lat, lon, level_density = 6):
    stream_plot = plt.streamplot(lon, lat, u, v,
                                 density=level_density)
    plt.close()
    seen = set()
    unique_segments = []

    for segment in stream_plot.lines.get_segments():
        rounded = np.round(segment, decimals=0)
        key = rounded.tobytes()  # Fast hashable representation
        if key not in seen:
            seen.add(key)
            unique_segments.append(rounded)

    streamlines = build_streamlines(unique_segments)
    isolines_max = [line[:, 1].max() for line in streamlines]

    return streamlines, isolines_max


def eddy_already_detected(eddi: dict, lon_eddie: float, lat_eddie: float):
    for e in eddi:
        if eddi[e]["lon"] == lon_eddie and eddi[e]["lat"] == lat_eddie:
            return True
    return False


def detect_UV_core(data, det_param, U, V, SPEED, t, dxC, dyC):
    """
    Detect eddy cores according to criterions:
    1.a) reversal of direction in v and b) v increase "a" points away from reversal
    2.a) reversal of direction in u and b) u increase "a" points away from reversal
    3) Eddy center placed at the velocity minimum within the searching area "b" points
    around the points that satisfy the first two constraints
    4) A closed contour exist around the potential eddy center with a valid winding angle (Chaigneau 2008)

    Parameters
    ------
    data = data_aligned from preprocess_inputs()
    det_param from define_detection_parameter
    U = xarray Dataset of velocity U with coordinate time (data_aligned["UVEL"])
    V, SPEED,
    t=time index to select u = U.isel(time=t).values,
    dxC = data_aligned['dxC'].values,
    dyC = data_aligned['dyC'].values

    Returns
    -------
    dictionary eddi = {}
    """
    u = U.isel(time=t).values
    v = V.isel(time=t).values
    speed = SPEED.isel(time=t).values
    lon, lat = np.meshgrid(data.lon.sel(lon=slice(det_param["lon1"],
                                                  det_param["lon2"])).values,
                           data.lat.sel(lat=slice(det_param["lat1"],
                                                  det_param["lat2"])).values)
    a = det_param["a"]
    b = det_param["b"]
    rad = det_param["rad"]
    bounds = np.shape(speed)
    d_thres = 1000
    # initialise eddy counter & output dict
    e = 0
    eddi = {}
    
    for i in range(len(v[:, 0])-1):
        try:
            v_slice = v[i, :]

            # ------------
            # 1.a) reversal of direction in v
            s = np.sign(v_slice)
            index_v_reversal = np.where((np.diff(s) != 0) & (~np.isnan(np.diff(s))))[0]
            for ii in range(len(index_v_reversal)-1):
                idx_v = index_v_reversal[ii]

                # 1.b) v increase "a" points away from reversal
                v0 = v_slice[idx_v]
                v_next = v_slice[idx_v + 1]
                v_minus_a = v_slice[idx_v - a]
                v_plus_a = v_slice[idx_v + 1 + a]
                if (v0 >= 0) & (v_minus_a > v0) & (v_plus_a < v_next):
                    # anti-cyclonic
                   var = -1
                elif (v0 < 0) & (v_minus_a < v0) & (v_plus_a > v_next):
                    # cyclonic
                    var = 1
                else:
                    # 1.b) not fulfilled
                    continue

                # ------------
                # 2.a) reversal of direction in u and b) u increase "a" points away from reversal
                # anticyclonic
                if var == -1:
                    if ((u[i-a, idx_v] <= 0) & (u[i - a, idx_v] <= u[i - 1, idx_v])
                         & (u[i+a, idx_v] >= 0) & (u[i + a, idx_v] >= u[i + 1, idx_v])):
                        var = -1
                    elif ((u[i-a, idx_v + 1] <= 0) & (u[i - a, idx_v + 1] <= u[i - 1, idx_v + 1])
                           & (u[i+a, idx_v + 1] >= 0) & (u[i + a, idx_v + 1] >= u[i + 1, idx_v + 1])):
                        var = -1
                    else:
                        # 2. not fulfilled
                        continue
                # cyclonic
                elif var == 1:
                    if ((u[i-a, idx_v] >= 0) & (u[i - a, idx_v] >= u[i - 1, idx_v])
                         & (u[i+a, idx_v] <= 0) & (u[i + a, idx_v] <= u[i + 1, idx_v])):
                        var = 1
                    elif ((u[i-a, idx_v + 1] >= 0) & (u[i - a, idx_v + 1] >= u[i - 1, idx_v + 1])
                       & (u[i+a, idx_v + 1] <= 0) & (u[i + a, idx_v + 1] <= u[i + 1, idx_v + 1])):
                        var = 1
                    else:
                        # 2. not fulfilled
                        continue

                # ------------
                # 3) Eddy center placed at the velocity minimum within the searching area "b" points
                # around the points that satisfy the first two constraints

                # find the velocity minimum within the searching area defined by
                # "b" around the potential eddy center

                # velocity magnitude, latitude and longitude within the searching area
                srch = speed[i-b:i+b, idx_v - b:idx_v + 1 + b]
                slat = lat[i-b:i+b, idx_v - b:idx_v + 1 + b]
                slon = lon[i-b:i+b, idx_v - b:idx_v + 1 + b]
                # position of the velocity minimum within the searching area
                idx_x_eddy, idx_y_eddy = np.where(srch == np.nanmin(srch))

                if len(idx_x_eddy) != 1:
                    idx_x_eddy=idx_x_eddy[0]
                    idx_y_eddy=idx_y_eddy[0]

                if eddy_already_detected(eddi, slon[idx_x_eddy, idx_y_eddy], slat[idx_x_eddy, idx_y_eddy]):
                    continue

                # second searching area centered around the velocity minimum
                srch2 = speed[int(max((i-b)+(idx_x_eddy-1)-b, 0)):int(min((i-b)+(idx_x_eddy-1)+b, len(speed[:,0]))),
                        int(max((idx_v - b) + (idx_y_eddy - 1) - b, 0)):int(min((idx_v - b) + (idx_y_eddy - 1) + b, len(speed[0,:])))]
                # if the two minima coincide then it is a local minima
                if (np.nanmin(srch2) != np.nanmin(srch)):
                    continue

                # ------------
                # 4) A closed contour exist around the potential eddy center with a valid winding angle (Chaigneau 2008)
                # check the rotation of the vectors along the boundary of the area
                # "a-1" around the points which satisfy the first three constraints
                # indices of the estimated center in the large domain
                i1, i2 = np.where((lat == slat[idx_x_eddy, idx_y_eddy]) & (lon == slon[idx_x_eddy, idx_y_eddy]))

                u_large = u[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                            int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
                v_large = v[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                            int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
                speed_large = speed[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                                    int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
                lon_large = lon[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                                int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
                lat_large = lat[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                                int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
                dxC_large = dxC[int(max(i1 - (rad * a), 1)):int(min(i1 + (rad * a), bounds[0])),
                                int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
                dyC_large = dyC[int(max(i1 - (rad * a), 1)):int(min(i1 + (rad * a), bounds[0])),
                                int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]

                eddy_i, eddy_j, eddy_mask, winding, isolines = get_eddy_contour(u_large,
                                                                                v_large,
                                                                                lon_large,
                                                                                lat_large,
                                                                                slon[idx_x_eddy, idx_y_eddy],
                                                                                slat[idx_x_eddy, idx_y_eddy],
                                                                                d_thres)

                if eddy_i is None:
                    #debug
                    eddi[e] = {}
                    eddi[e]['valid_eddy'] = -1
                    eddi[e]['winding'] = winding
                    eddi[e]["lon"] = slon[idx_x_eddy, idx_y_eddy]
                    eddi[e]["lat"] = slat[idx_x_eddy, idx_y_eddy]
                    eddi[e]['time'] = U.isel(time=t).time.values
                    eddi[e]['isolines'] = isolines
                    e += 1
                    continue

                # ------------
                # All criterions are satisfied

                eddi[e] = {}
                eddi[e]['valid_eddy'] = 1
                eddi[e]["lon"] = slon[idx_x_eddy, idx_y_eddy]
                eddi[e]["lat"] = slat[idx_x_eddy, idx_y_eddy]
                j_min = (data.lat.where(data.lat == np.nanmin(lat_large), other=0) ** 2).argmax().values
                i_min = (data.lon.where(data.lon == np.nanmin(lon_large), other=0) ** 2).argmax().values
                eddi[e]['eddy_j'] = eddy_j + j_min
                eddi[e]['eddy_i'] = eddy_i + i_min
                eddi[e]['time'] = U.isel(time=t).time.values
                area = (dxC_large / 1000.) * (dyC_large / 1000.) * eddy_mask
                eddi[e]['area'] = np.array([np.nansum(area)])
                eddi[e]['scale'] = np.array([np.sqrt(eddi[e]['area'] / np.pi)])
                eddi[e]['winding'] = winding
                if det_param["hemi"] == "north":
                    if var == -1:
                        eddi[e]['type'] = "anticyclonic"
                    elif var ==1:
                        eddi[e]['type'] = "cyclonic"
                elif det_param["hemi"] == "south":
                    if var == -1:
                        eddi[e]['type'] = "cyclonic"
                    elif var ==1:
                        eddi[e]['type'] = "anticyclonic"
                e += 1
        except Exception as e:
            continue

    return eddi


def check_input_validity(use_bags, use_mp, det_param, data):
    # make sure arguments are compatible
    if use_bags and use_mp:
        raise ValueError('Cannot use dask_bags and multiprocessing at the'
                         + 'same time. Set either `use_bags` or `use_mp`'
                         + 'to `False`.')
    # Verify that the specified region lies within the dataset provided
    if (det_param['lon1'] < np.around(data['lon'].min())
        or det_param['lon2'] > np.around(data['lon'].max())):
        raise ValueError('`det_param`: min. and/or max. of longitude range'
                         + ' are outside the region contained in the dataset')
    if (det_param['lat1'] < np.around(data['lat'].min())
        or det_param['lat2'] > np.around(data['lat'].max())):
        raise ValueError('`det_param`: min. and/or max. of latitude range'
                         + ' are outside the region contained in the dataset')


def define_start_and_end_dates(det_param, data):
    if det_param['calendar'] == 'standard':
        start_time = np.datetime64(det_param['start_time'])
        end_time = np.datetime64(det_param['end_time'])
    elif det_param['calendar'] == '360_day':
        start_time = cft.Datetime360Day(int(det_param['start_time'][0:4]),
                                        int(det_param['start_time'][5:7]),
                                        int(det_param['start_time'][8:10]))
        end_time = cft.Datetime360Day(int(det_param['end_time'][0:4]),
                                      int(det_param['end_time'][5:7]),
                                      int(det_param['end_time'][8:10]))
    else:
        raise ValueError('Wrong calendar type. Should be "standard" or "360_day"')
    if (start_time > data['time'][-1]
        or end_time < data['time'][0]):
        raise ValueError('`det_param`: there is no overlap of the original time'
                         + ' axis and the desired time range for the'
                         + ' detection')

    return start_time, end_time


def detect_UV(data, det_param, u_var, v_var, speed_var, use_bags=False, use_mp=False, 
              mp_cpu=2, regrid_avoided=False):
    # Checking that inputs make sense
    check_input_validity(use_bags, use_mp, det_param, data)
    # Defining useful variables
    start_time, end_time = define_start_and_end_dates(det_param, data)
    U = data[u_var].compute()
    V = data[v_var].compute()
    SPEED = data[speed_var].compute()
    dxC = data['dxC'].values
    dyC = data['dyC'].values
    
    if use_mp:
        ## set range of parallel executions
        pexps = range(0, len(U['time']))
        ## prepare arguments
        arguments = zip(
                        itertools.repeat(data),
                        itertools.repeat(det_param.copy()),
                        itertools.repeat(U),
                        itertools.repeat(V),
                        itertools.repeat(SPEED),
                        pexps,
                        itertools.repeat(dxC),
                        itertools.repeat(dyC)
                        )
        print("Detecting eddies in velocity fields")
        if mp_cpu > mp.cpu_count():
            mp_cpu = mp.cpu_count()
        with mp.Pool(mp_cpu) as p:
            eddies = p.starmap(detect_UV_core, arguments)
        p.close()
        p.join()
    elif use_bags:
        ## set range of parallel executions
        pexps = range(0, len(U['time']))
        ## generate dask bag instance
        seeds_bag = dask_bag.from_sequence(pexps)
        print("Detecting eddies in velocity fields")
        detection = dask_bag.map(
            lambda tt: detect_UV_core(data, det_param.copy(), U, V, SPEED, tt,
                                      dxC, dyC)
                                 ,seeds_bag)
        eddies = detection.compute()
    else:
        eddies = {}
        for tt in np.arange(0, len(U['time'])):
            steps = np.around(np.linspace(0, len(U['time']), 10))
            if tt in steps:
                print('detection at time step ', str(tt + 1), ' of ',
                      len(U['time']))
            eddies[tt] = detect_UV_core(data, det_param.copy(),
                                        U, V, SPEED, tt, dxC, dyC)
    return eddies
