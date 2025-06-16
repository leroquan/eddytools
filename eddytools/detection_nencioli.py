'''detection
Collection of functions needed for the detection of mesoscale eddies
based on the Nencioli (2010) method. The data is assumed to have been
interpolated with the `interp.py` module of this package or at least
needs to have the same structure.
'''

import numpy as np
import pandas as pd
from matplotlib.path import Path
import xarray as xr
from scipy import ndimage
from scipy.signal import find_peaks
import cftime as cft
import itertools
from scipy import interpolate
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
    data_cropped = data_cropped.isel(Z=0)

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


def inpolygon(point_i, point_j, x, y):
    p = np.array([x, y]).T
    hullp = ConvexHull(p)
    poly_path = Path(p[hullp.vertices])
    return poly_path.contains_point([point_i, point_j])


def inpolygon2D(points_i, points_j, x, y):
    p = np.array([x, y]).T
    hullp = ConvexHull(p)
    poly_path = Path(p[hullp.vertices, :])
    return poly_path.contains_points(np.array([points_i, points_j]).T)


def interpolate_limits(pt_i, pt_j, lon, lat, dkm, vel, direct):
    # include only 'n' points around that point to interpolate vel
    # (griddata is a very slow process) --------------------------------
    # find the closest grid point to the curve extreme
    dist = np.sqrt((lon - pt_i)**2 + (lat - pt_j)**2)
    d_j, d_i = np.where(dist == np.min(dist))
    d_j, d_i =  d_j[0], d_i[0] 
    n = 4 # at least 4 points away to avoid a qhull precision warning!!!
    # resize coordinate and velocity matrices
    svel = vel[int(np.max([d_j-n, 1])):int(np.min([d_j+n, np.shape(vel)[0]])),
               int(np.max([d_i-n, 1])):int(np.min([d_i+n, np.shape(vel)[1]]))]
    skm_i = lon[int(np.max([d_j-n, 1])):int(np.min([d_j+n, np.shape(vel)[0]])),
                int(np.max([d_i-n, 1])):int(np.min([d_i+n, np.shape(vel)[1]]))]
    skm_j = lat[int(np.max([d_j-n, 1])):int(np.min([d_j+n, np.shape(vel)[0]])),
                int(np.max([d_i-n, 1])):int(np.min([d_i+n, np.shape(vel)[1]]))]
    # interpolate vel across the curve along different directions depending on
    # different extremes
    if direct == "N":
        pts = np.array([[pt_i, float(pt_j-dkm)], [pt_i, float(pt_j+dkm)]])
    elif direct == "S":
        pts = np.array([[pt_i, float(pt_j+dkm)], [pt_i, float(pt_j-dkm)]])
    elif direct == "W":
        pts = np.array([[float(pt_i+dkm), pt_j], [float(pt_i-dkm), pt_j]])
    elif direct == "E":
        pts = np.array([[float(pt_i-dkm), pt_j], [float(pt_i+dkm), pt_j]])
    # interpolate vel
    in_out_vel = interpolate.griddata(np.array([skm_i.flatten(), skm_j.flatten()]).T,
                                      svel.flatten(), pts)
    return in_out_vel


def get_eddy_indeces(lon, lat, ec_lon, ec_lat, psi, vel):
    # define the distance from the contour extremes at which velocity magnitude
    # is interpolated
    (c_j, c_i) = np.where((lat==ec_lat) & (lon==ec_lon))
    dkm = 0.05 * (lon[c_j, 1] - lon[c_j, 0])
    # compute contourlines of the streamfunction field (100 contours)
    cg = contour_generator(lon, lat, psi)
    C = [cg.lines(np.linspace(np.nanmin(psi), np.nanmax(psi), 100)[i]) for i in np.arange(100)]
    
    # intialize the two variables
    eddy_lim = []
    largest_curve=[]
    # rearrange all the contours in C to the structure array 'isolines'
    # each element of isolines contains all the vertices of a given contour
    # level of PSI
    isolines = {}
    isolines_max = []
    i = 0
    ii = 0
    there_are_contours = True
    while there_are_contours:
        for j in np.arange(len(C[ii])):
            isolines[i] = {}
            isolines[i]["x"] = []
            isolines[i]["y"] = []
            isolines[i]["x"] = list(C[ii][j][:, 0])
            isolines[i]["y"] = list(C[ii][j][:, 1])
            isolines_max.append(np.nanmax(isolines[i]["y"]))
            i += 1
        ii += 1
        try:
            test = C[ii]
            there_are_contours = True
        except:
            there_are_contours = False
    # sort the contours accroding to their maximum latitude; this way the first
    # closed contour across which velocity increases will also be the largest
    # one (it's the one which extend further north).
    sorted_iso = np.argsort(isolines_max)[::-1]
    #print(f"lenght iso: {len(sorted_iso)}")
    
    # restart the counter and initialize the two flags
    i = 0
    closed_indx = 0 # set to 1 when eddy shape is found
    largest_indx = 0 # set to 1 when largest closed contour is found
    # inspect all isolines until the eddy shape is determined
    # (closed_indx=1 stops the loop)
    while ((closed_indx == 0) & (i < len(isolines))):
        ii = sorted_iso[i]
        xdata = isolines[ii]["x"] # vertex lon's
        ydata = isolines[ii]["y"] # vertex lat's
        # conditions to have the largest closed contour around the center
        # (isolines already sorted by maximum latitude)
        # 1) closed contours
        # 2) detected eddy center inside the polygon
        if ((len(xdata) < 3) or (len(ydata) < 3)):
            i += 1
            continue
        try:
            inpo = inpolygon(ec_lon, ec_lat, xdata, ydata)
        except Exception as e:
            #print(f"ec_lon={ec_lon}, ec_lat={ec_lat}")
            #print('Error in inpolygon')
            #print(f'Error: {e}')
            inpo = False
        if (((xdata[0] == xdata[-1]) & (ydata[0] == ydata[-1])) & inpo):
            #print('Contour is closed.')
            # find the contour extremes
            Nj = np.max(ydata)
            Ni = np.max(xdata[int(np.where(ydata==Nj)[0][0])])
            Sj = np.min(ydata)
            Si = np.min(xdata[int(np.where(ydata==Sj)[0][0])])
            Ei = np.max(xdata)
            Ej = np.min(ydata[int(np.where(xdata==Ei)[0][0])])
            Wi = np.min(xdata)
            Wj = np.max(ydata[int(np.where(xdata==Wi)[0][0])])
            # check if velocity across the contour increases
            direct = ['N', 'S', 'E', 'W']
            pts_I = [Ni, Si, Ei, Wi]
            pts_J = [Nj, Sj, Ej, Wj]
            # inspect one extreme at the time (faster)
            iii = 0 # counter
            smaller_vel = 0  # flag to stop the loop (1 if velocity decreases
                             # across the fourth extremes)
            smaller_vel1 = 0 # flag to stop the loop (1 if velocity decreases
                             # across the third extremes)
            smaller_vel2 = 0 # flag to stop the loop (1 if velocity decreases
                             # across the second extremes)
            smaller_vel3 = 0 # flag to stop the loop (1 if velocity decreases
                             # across the first extremes)
            while ((iii < len(direct)) & (smaller_vel == 0)):
                # interpolate velocity across the extreme
                in_out_vel = interpolate_limits(pts_I[iii], pts_J[iii],
                                                lon, lat, dkm, vel, direct[iii])
                # change the flag value if velocity decreases
                if (in_out_vel[0] > in_out_vel[1]):
                    if (smaller_vel3 == 0):
                        smaller_vel3 = 1
                    elif (smaller_vel2 == 0):
                        smaller_vel2 = 1
                    elif (smaller_vel1 == 0):
                        smaller_vel1 = 1
                    elif (smaller_vel == 0):
                        smaller_vel = 1
                iii += 1 # increase the counter
            # only if velocity increases across all four extremes the closed
            # contour is saved as eddy shape
            if (smaller_vel == 0):
                eddy_lim = [xdata, ydata]
                closed_indx = 1
            # largest closed conotur is saved as well
            if (largest_indx == 0):
                largest_curve = [xdata, ydata]
                largest_indx = 1
        i += 1 # increase the counter
    # in case velocity doesn't increase across the closed contour, eddy shape
    # is defined simply as the largest closed contour
    if ((eddy_lim==[]) & (largest_curve!=[])):
        eddy_lim = largest_curve
    if eddy_lim==[]:
        return None, None, None
    else:
        mask = inpolygon2D(lon.flatten(), lat.flatten(), eddy_lim[0], eddy_lim[1])
        eddy_mask = mask.reshape(np.shape(lon))
        eddy_j = np.where(eddy_mask)[0]
        eddy_i = np.where(eddy_mask)[1]
        return eddy_i, eddy_j, eddy_mask


def eddi_exists(eddi, lon_eddie, lat_eddie):
    for e in eddi:
        if eddi[e]["lon"] == lon_eddie and eddi[e]["lat"] == lat_eddie:
            return True
    return False


def detect_UV_core(data, det_param, U, V, SPEED, t, e1f, e2f,
                   regrid_avoided=False):
    if regrid_avoided == True:
        raise ValueError("regrid_avoided cannot be used (yet).")
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
    borders = np.max([a, b]) + 1
    bounds = np.shape(speed)
    # initialise eddy counter & output dict
    e = 0
    eddi = {}
    
    for i in np.arange(borders, len(v[:, 0])-borders+1):
        wrk = v[i, :]
        # reversal of direction in V
        s = np.sign(wrk)
        indx = np.where((np.diff(s) != 0) & (~np.isnan(np.diff(s))))[0]
        indx = indx[((indx > borders) & (indx < (len(wrk) - borders)))]
        for ii in np.arange(0, len(indx)):
            # v increase "a" points away from reversal
            # anticyclonic
            if wrk[indx[ii]] >= 0:
                if ((wrk[indx[ii]-a] > wrk[indx[ii]])
                    & (wrk[indx[ii]+1+a] < wrk[indx[ii]+1])):
                    var = -1
                else:
                    var = 0
            # cyclonic
            elif wrk[indx[ii]] < 0:
                if ((wrk[indx[ii]-a] < wrk[indx[ii]])
                    & (wrk[indx[ii]+1+a] > wrk[indx[ii]+1])):
                    var = 1
                else:
                    var = 0
            # reversal of direction in U and increase away from reversal
            # anticyclonic
            if var == -1:
                if (((u[i-a, indx[ii]] <= 0) & (u[i-a, indx[ii]] <= u[i-1, indx[ii]])
                     & (u[i+a, indx[ii]] >= 0) & (u[i+a, indx[ii]] >= u[i+1, indx[ii]]))
                    | ((u[i-a, indx[ii]+1] <= 0) & (u[i-a, indx[ii]+1] <= u[i-1, indx[ii]+1])
                       & (u[i+a, indx[ii]+1] >= 0) & (u[i+a, indx[ii]+1] >= u[i+1, indx[ii]+1]))):
                    var = -1
                    #eddy_uv.append([(lat[i, indx[ii]], lon[i, indx[ii]]),
                    #                (lat[i, indx[ii]+1], lon[i, indx[ii]+1])])
                else:
                    var=0
            # cyclonic
            elif var == 1:
                if (((u[i-a, indx[ii]] >= 0) & (u[i-a, indx[ii]] >= u[i-1, indx[ii]])
                     & (u[i+a, indx[ii]] <= 0) & (u[i+a, indx[ii]] <= u[i+1, indx[ii]]))
                    | ((u[i-a, indx[ii]+1] >= 0) & (u[i-a, indx[ii]+1] >= u[i-1, indx[ii]+1])
                       & (u[i+a, indx[ii]+1] <= 0) & (u[i+a, indx[ii]+1] <= u[i+1, indx[ii]+1]))):
                    var = 1
                    #eddy_uv.append([(lat[i, indx[ii]], lon[i, indx[ii]]),
                    #                (lat[i, indx[ii]+1], lon[i, indx[ii]+1])])
                else:
                    var=0
                    
            # find the velocity minimum within the searching area defined by
            # "b" around the points that satisfy the first two constraints
            if var != 0:
                
                #print(f'u or v increases "a={a}" points away from point of index ({i},{ii}). Searching for minima in {b} cells around...')
                
                # velocity magnitude, latitude and longitude within the
                # searching area
                srch = speed[i-b:i+b, indx[ii]-b:indx[ii]+1+b]
                slat = lat[i-b:i+b, indx[ii]-b:indx[ii]+1+b]
                slon = lon[i-b:i+b, indx[ii]-b:indx[ii]+1+b]
                # position of the velocity minimum within the searching area
                X, Y = np.where(srch == np.nanmin(srch))                
                
                if len(X) == 1 and not eddi_exists(eddi, slon[X, Y], slat[X, Y]):
                    # second searching area centered around the velocity minimum
                    # (bound prevents this area from extending outside the domain)
                    srch2 = speed[int(max((i-b)+(X-1)-b, 1)):int(min((i-b)+(X-1)+b, bounds[0])),
                                  int(max((indx[ii]-b)+(Y-1)-b, 1)):int(min((indx[ii]-b)+(Y-1)+b, bounds[1]))]
                    # if the two minima coincide then it is a local minima
                    if (np.nanmin(srch2) != np.nanmin(srch)):
                        var = 0
                    #else:
                    #    eddy_c.append([(slat[X, Y][0], slon[X, Y][0])])
                else:
                    var = 0
                
            # check the rotation of the vectors along the boundary of the area
            # "a-1" around the points which satisfy the first three constraints
            d = a - 1
            if var != 0:
                # indices of the estimated center in the large domain
                i1, i2 = np.where((lat == slat[X, Y]) & (lon == slon[X, Y]))
                
                #print(f'Local minima found in indices ({i1},{i2}). Looking for rotation...')
                
                # velocities within "a-1" points from the estimated center
                u_small = u[int(max(i1-d, 1)):int(min(i1+d+1, bounds[0])),
                            int(max(i2-d, 1)):int(min(i2+d+1, bounds[1]))]
                v_small = v[int(max(i1-d, 1)):int(min(i1+d+1, bounds[0])),
                            int(max(i2-d, 1)):int(min(i2+d+1, bounds[1]))]
                lon_small = lon[int(max(i1-d, 1)):int(min(i1+d+1, bounds[0])),
                            int(max(i2-d, 1)):int(min(i2+d+1, bounds[1]))]
                lat_small = lat[int(max(i1-d, 1)):int(min(i1+d+1, bounds[0])),
                            int(max(i2-d, 1)):int(min(i2+d+1, bounds[1]))]
                # constraint is applied only if sea-points are within the area
                if ~np.isnan(u_small).all():
                    # boundary velocities
                    u_bound = [u_small[0, :], u_small[1::, -1],
                               u_small[-1, -2:0:-1], u_small[-1:0:-1, 0]]
                    u_bound = np.array([item for sublist in u_bound for item in sublist])
                    v_bound = [v_small[0, :], v_small[1::, -1],
                               v_small[-1, -2:0:-1], v_small[-1:0:-1, 0]]
                    v_bound = np.array([item for sublist in v_bound for item in sublist])
                    # vector defining which quadrant each boundary vector
                    # belongs to
                    quadrants = np.zeros_like(u_bound)
                    quadrants[((u_bound >= 0) & (v_bound >= 0))] = 1
                    quadrants[((u_bound < 0) & (v_bound >= 0))] = 2
                    quadrants[((u_bound < 0) & (v_bound < 0))] = 3
                    quadrants[((u_bound >= 0) & (v_bound < 0))] = 4
                    # used identify which is the firts fourth quadrant vector
                    spin = np.where(quadrants==4)[0]
                    # apply the constraint only if complete rotation and not
                    # all vectors in the fourth quadrant
                    if ((spin.size != 0) & (spin.size != quadrants.size)):
                        
                        #print(f"There is complete rotation !! Checking if it's uniform...")
                        
                        # if vectors start in 4 quadrant, then I add 4 to all
                        # quandrant positions from the first 1 occurrence
                        if spin[0] == 0:
                            spin = np.where(quadrants!=4)[0]
                            spin = spin[0] - 1
                        else:
                            spin = spin[-1]
                        quadrants[spin+1::] = quadrants[spin+1::] + 4
                        # inspect vector rotation:
                        # - no consecutive vectors more than one quadrant away
                        # - no backward rotation
                        if ((np.where(np.diff(quadrants) > 1)[0].size == 0)
                            & (np.where(np.diff(quadrants) < 0)[0].size == 0)):                            
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
                            eddy_i, eddy_j, eddy_mask = get_eddy_indeces(lon_large, lat_large,
                                                                         slon[X, Y][0], slat[X, Y][0], psi, speed_large)

                            if ((np.shape(eddy_i)!=()) & (np.shape(eddy_j)!=()) & (np.shape(eddy_mask)!=())):
                                eddi[e] = {}
                                eddi[e]["lon"] = slon[X, Y]
                                eddi[e]["lat"] = slat[X, Y]
                                j_min = (data.lat.where(data.lat == np.nanmin(lat_large), other=0) ** 2).argmax().values
                                i_min = (data.lon.where(data.lon == np.nanmin(lon_large), other=0) ** 2).argmax().values
                                eddi[e]['eddy_j'] = eddy_j + j_min
                                eddi[e]['eddy_i'] = eddy_i + i_min
                                eddi[e]['time'] = U.isel(time=t).time.values
                                eddi[e]['amp'] = np.array([np.nanmax(psi * eddy_mask) - np.nanmin(psi * eddy_mask)])
                                area = (e1f_large / 1000.) * (e2f_large / 1000.) * eddy_mask
                                eddi[e]['area'] = np.array([np.nansum(area)])
                                eddi[e]['scale'] = np.array([np.sqrt(eddi[e]['area'] / np.pi)])
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
    if (start_time > data['time'][-1]
        or end_time < data['time'][0]):
        raise ValueError('`det_param`: there is no overlap of the original time'
                         + ' axis and the desired time range for the'
                         + ' detection')

    return (start_time, end_time)


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
