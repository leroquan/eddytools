'''detection
Collection of functions needed for the detection of mesoscale eddies
based on the Nencioli (2010) method. The data is assumed to have been
interpolated with the `interp.py` module of this package or at least
needs to have the same structure.
'''

import numpy as np
from matplotlib.path import Path
import xarray as xr
import cftime as cft
import itertools
from contourpy import contour_generator
from scipy.spatial import ConvexHull
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


def compute_psi(u_psi, v_psi, dx, dy):
    # compute PSI to get eddy area
    # Indices for domain size
    lx = np.shape(u_psi)[0]
    ly = np.shape(u_psi)[1]
    # itegrate first row of v along longitude (first term of eq.A2)
    cx = np.nancumsum(v_psi[0, :]) * dx[0, :]
    # integrate first column of u along latitude (second term of eq.A3)
    cy = np.nancumsum(u_psi[:, 0]) * dy[:, 0]
    # compute streamfunction
    # PSI from integrating v firts and then u
    psi_xy = (-cx[None, :] + np.nancumsum(u_psi, axis=0) * dy)
    # PSI from integrating u first and then v
    psi_yx = (-np.nancumsum(v_psi, axis=1) * dx + cy[:, None])
    # final PSI as average between the two
    psi = (psi_xy + psi_yx) / 2

    return np.where(np.isnan(u_psi), np.nan, psi)


def check_if_in_polygon(x_center: float, y_center: float, x_isoline, y_isoline):
    p = np.array([x_isoline, y_isoline]).T
    hullp = ConvexHull(p)
    poly_path = Path(p[hullp.vertices])

    return poly_path.contains_point((x_center, y_center))


def inpolygon2D(points_i, points_j, x, y):
    p = np.array([x, y]).T
    hullp = ConvexHull(p)
    poly_path = Path(p[hullp.vertices, :])
    return poly_path.contains_points(np.array([points_i, points_j]).T)


def normalize_angle(angle):
    """Normalize angle to [-180, 180] degrees."""
    angle = (angle + 180) % 360 - 180
    return angle


def check_winding_angle(stline_x, stline_y, winding_thres=360, baddir_thres=90, d_thres=5):
    """
    Detects eddies in a streamline by analyzing its winding angle.

    Parameters:
        stline_x (array-like): X coordinates of the streamline.
        stline_y (array-like): Y coordinates of the streamline.
        winding_thres (float): Threshold winding angle to declare an eddy (degrees).
        baddir_thres (float): Threshold for angle deviation in wrong direction (degrees).
        d_thres (float): Distance threshold for closed loop detection.

    Returns:
        (bool, float): (eddy_detected, total_winding_angle)
    """
    winding = 0.0
    baddir = 0.0
    dir_sign = 0
    eddy = False
    min_dist = float('inf')

    # Initial direction
    dx0 = stline_x[1] - stline_x[0]
    dy0 = stline_y[1] - stline_y[0]
    prev_angle = np.degrees(np.arctan2(dy0, dx0))

    for i in range(1, len(stline_x) - 1):
        angle_diff, winding = increment_winding_angle(i, prev_angle, stline_x, stline_y, winding)

        if not check_direction_consistency(angle_diff, baddir, baddir_thres, dir_sign):
            break

        # Eddy detection logic
        if abs(winding) > winding_thres:
            dist_to_start = np.hypot(stline_x[i] - stline_x[0], stline_y[i] - stline_y[0])
            if eddy and dist_to_start > min_dist:
                break
            if dist_to_start < d_thres:
                eddy = True
                min_dist = dist_to_start

    return eddy, winding


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
    dx = stline_x[i + 1] - stline_x[i]
    dy = stline_y[i + 1] - stline_y[i]
    curr_angle = np.degrees(np.arctan2(dy, dx))
    angle_diff = normalize_angle(curr_angle - prev_angle)
    prev_angle = curr_angle
    winding += angle_diff

    return angle_diff, winding


def get_eddy_contour(lon, lat, ec_lon, ec_lat, psi):
    """
    Get the largest contour around eddy center fitting these criterions:
    1) detected eddy center inside the polygon
    2) closed contour
    3) valid winding angle

    Returns
    -------
    eddy_i, eddy_j, eddy_mask, winding, isolines
    """
    isolines, isolines_max = generate_contours(lat, lon, psi)

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
    winding=0
    for i in range(len(isolines)):
        ii = sorted_iso[i]
        xdata = isolines[ii]["x"] # vertex lon's
        ydata = isolines[ii]["y"] # vertex lat's
        winding=0

        # 1) detected eddy center inside the polygon
        try:
            is_in_polygon = check_if_in_polygon(ec_lon, ec_lat, xdata, ydata)
        except Exception as e:
            is_in_polygon = False
        if not is_in_polygon:
            continue

        # 2) closed contour
        if not ((xdata[0] == xdata[-1]) & (ydata[0] == ydata[-1])):
            continue

        # 3) valid winding angle
        valid_winding, winding = check_winding_angle(xdata, ydata, winding_thres=340, baddir_thres=90, d_thres=5)
        if valid_winding:
            # Found the eddy contour
            eddy_lim = [xdata, ydata]
            break

    eddy_i, eddy_j, eddy_mask =  None, None, None
    if eddy_lim:
        mask = inpolygon2D(lon.flatten(), lat.flatten(), eddy_lim[0], eddy_lim[1])
        eddy_mask = mask.reshape(np.shape(lon))
        eddy_j = np.where(eddy_mask)[0]
        eddy_i = np.where(eddy_mask)[1]

    return eddy_i, eddy_j, eddy_mask, winding, isolines


def generate_contours(lat, lon, psi):
    # Generate contour levels
    levels = np.linspace(np.nanmin(psi), np.nanmax(psi), 200)
    # Generate contours
    cg = contour_generator(lon, lat, psi)
    contours = [cg.lines(level) for level in levels]
    # Initialize structures
    isolines = {}
    isolines_max = []
    # Flatten and store contour line vertices
    idx = 0
    for contour_level in contours:
        for segment in contour_level:
            if len(segment[:, 0]) > 3:
                isolines[idx] = {
                    "x": segment[:, 0].tolist(),
                    "y": segment[:, 1].tolist()
                }
            isolines_max.append(np.nanmax(segment[:, 1]))
            idx += 1
    return isolines, isolines_max


def eddy_already_detected(eddi, lon_eddie, lat_eddie):
    for e in eddi:
        if eddi[e]["lon"] == lon_eddie and eddi[e]["lat"] == lat_eddie:
            return True
    return False


def detect_UV_core(data, det_param, U, V, SPEED, t, e1f, e2f):
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
    e1f = data_aligned['dxC'].values,
    e2f = data_aligned['dyC'].values

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
    # initialise eddy counter & output dict
    e = 0
    eddi = {}
    
    for i in range(len(v[:, 0])):
        v_slice = v[i, :]

        # ------------
        # 1.a) reversal of direction in v
        s = np.sign(v_slice)
        index_v_reversal = np.where((np.diff(s) != 0) & (~np.isnan(np.diff(s))))[0]
        for ii in range(len(index_v_reversal)):
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

            if eddy_already_detected(eddi, slon[idx_x_eddy, idx_y_eddy], slat[idx_x_eddy, idx_y_eddy]):
                continue
            if len(idx_x_eddy) != 1:
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
            e1f_large = e1f[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                            int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
            e2f_large = e2f[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                            int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]

            psi = compute_psi(u_large, v_large, e1f_large, e2f_large)
            eddy_i, eddy_j, eddy_mask, winding = get_eddy_contour(lon_large,
                                                                  lat_large,
                                                                  slon[idx_x_eddy, idx_y_eddy][0],
                                                                  slat[idx_x_eddy, idx_y_eddy][0],
                                                                  psi)

            if eddy_i is None:
                continue

            # ------------
            # All criterions are satisfied

            eddi[e] = {}
            eddi[e]["lon"] = slon[idx_x_eddy, idx_y_eddy]
            eddi[e]["lat"] = slat[idx_x_eddy, idx_y_eddy]
            j_min = (data.lat.where(data.lat == np.nanmin(lat_large), other=0) ** 2).argmax().values
            i_min = (data.lon.where(data.lon == np.nanmin(lon_large), other=0) ** 2).argmax().values
            eddi[e]['eddy_j'] = eddy_j + j_min
            eddi[e]['eddy_i'] = eddy_i + i_min
            eddi[e]['time'] = U.isel(time=t).time.values
            eddi[e]['amp'] = np.array([np.nanmax(psi * eddy_mask) - np.nanmin(psi * eddy_mask)])
            area = (e1f_large / 1000.) * (e2f_large / 1000.) * eddy_mask
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
    e1f = data['dxC'].values
    e2f = data['dyC'].values
    
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
                        itertools.repeat(e1f),
                        itertools.repeat(e2f)
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
                                      e1f, e2f)
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
                                        U, V, SPEED, tt, e1f, e2f)
    return eddies
