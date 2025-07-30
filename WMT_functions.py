from scipy.stats import binned_statistic_2d, binned_statistic
from scipy.sparse.linalg import lsqr
from scipy.interpolate import interp2d
# from scipy.interpolate import interp2d, RectBivariateSpline
# from scipy.interpolate import RegularGridInterpolator

from scipy.sparse import coo_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

import numpy as np
import xarray as xr 
import SXBQ as sx

def mk_ts_trans_ds(dS, dT, S_range, T_range, time):

    S_bin = np.arange(S_range[0], S_range[1] + dS, dS)
    T_bin = np.arange(T_range[0], T_range[1] + dT, dT)
    S_bini = S_bin[0:-1] + dS/2
    T_bini = T_bin[0:-1] + dT/2 
    S_binii = S_bini[0:-1] + dS/2
    T_binii = T_bini[0:-1] + dT/2

    S_binj = np.arange(S_bin[0] - dS/2, S_bin[-1] + dS, dS)
    T_binj = np.arange(T_bin[0] - dT/2, T_bin[-1] + dT, dT)

    ts_vol_empty = np.zeros((len(T_bini), len(S_bini), len(time)))
    dvol_empty = np.zeros((len(T_bini), len(S_bini), len(time)-2))

    Gt_empty = np.zeros((len(T_binii), len(S_bini), len(time)-2))
    Gs_empty = np.zeros((len(T_bini), len(S_binii), len(time)-2))

    Et_empty = np.zeros((len(T_bin), len(S_bini), len(time)-2))
    Es_empty = np.zeros((len(T_bini), len(S_bin), len(time)-2))

    coords = {'T_bin': T_bin, 'S_bin': S_bin, 
              'T_bini': T_bini, 'S_bini': S_bini, 
              'T_binii': T_binii, 'S_binii': S_binii,
              'T_binj': T_binj, 'S_binj': S_binj,
              'time': time,
              'time_mid': time[1:-1],
              'dT': dT, 'dS': dS}

    ts_trans = xr.Dataset(
            {"ts_vol": (['T_bini', 'S_bini', 'time'], ts_vol_empty.copy()),
             "ts_count": (['T_bini', 'S_bini', 'time'], ts_vol_empty.copy()),
             "ts_area_boundary": (['T_bini', 'S_bini', 'time'], ts_vol_empty.copy()),
             "ts_lat": (['T_bini', 'S_bini', 'time'], ts_vol_empty.copy()),
             "ts_lon": (['T_bini', 'S_bini', 'time'], ts_vol_empty.copy()),
             "ts_oxygen_concentration": (['T_bini', 'S_bini', 'time'], ts_vol_empty.copy()),
             "ts_aou": (['T_bini', 'S_bini', 'time'], ts_vol_empty.copy()),
             "ts_T": (['T_bini', 'S_bini', 'time'], ts_vol_empty.copy()),
             "ts_S": (['T_bini', 'S_bini', 'time'], ts_vol_empty.copy()),
             "ts_dep": (['T_bini', 'S_bini', 'time'], ts_vol_empty.copy()),
             "dvol_G": (['T_bini', 'S_bini', 'time_mid'], dvol_empty.copy()),
             "dvol_E": (['T_bini', 'S_bini', 'time_mid'], dvol_empty.copy()),
             "dvol_M": (['T_bini', 'S_bini', 'time_mid'], dvol_empty.copy()),
             "dvol_F": (['T_bini', 'S_bini', 'time_mid'], dvol_empty.copy()),
             "G_t": (['T_binii', 'S_bini', 'time_mid'], Gt_empty.copy()),
             "G_s": (['T_bini', 'S_binii', 'time_mid'], Gs_empty.copy()),
             "E_t": (['T_bin', 'S_bini', 'time_mid'], Et_empty.copy()),
             "E_s": (['T_bini', 'S_bin', 'time_mid'], Es_empty.copy()),
             "F_t": (['T_binii', 'S_bini', 'time_mid'], Gt_empty.copy()),
             "F_s": (['T_bini', 'S_binii', 'time_mid'], Gs_empty.copy()),
             "M_t": (['T_binii', 'S_bini', 'time_mid'], Gt_empty.copy()),
             "M_s": (['T_bini', 'S_binii', 'time_mid'], Gs_empty.copy()),
             "G_ti": (['T_bini', 'S_bini', 'time_mid'], dvol_empty.copy()),
             "G_si": (['T_bini', 'S_bini', 'time_mid'], dvol_empty.copy()),
             "E_ti": (['T_bini', 'S_bini', 'time_mid'], dvol_empty.copy()),
             "E_si": (['T_bini', 'S_bini', 'time_mid'], dvol_empty.copy()),
             "F_ti": (['T_bini', 'S_bini', 'time_mid'], dvol_empty.copy()),
             "F_si": (['T_bini', 'S_bini', 'time_mid'], dvol_empty.copy()),
             "M_ti": (['T_bini', 'S_bini', 'time_mid'], dvol_empty.copy()),
             "M_si": (['T_bini', 'S_bini', 'time_mid'], dvol_empty.copy()),
             "epsilon": (['T_bini', 'S_bini', 'time_mid'], dvol_empty.copy()),
             "lsq_relative_residual": (['time_mid'], np.ones(len(time[1:-1]))*np.nan)             
            },
            coords=coords)
    
    return ts_trans


def calc_ts_area(salinity, temperature, salinity_noflat, temperature_noflat, ts_trans, dx, dy, open_or_close):  ## m**2
    
    bin_edge1 = ts_trans.T_bin
    bin_edge2 = ts_trans.S_bin
    
    for ii in np.arange(0, len(ts_trans.time)):

        bin1_3D = temperature[ii]
        bin2_3D = salinity[ii]
        var_3D = bin1_3D.copy()*0+dx*1000*dy
        
        out = sx.grid2d(bin1_3D, bin2_3D, var_3D, 
                        xi=bin_edge1[:-1], yi=bin_edge2[:-1],
                        fn='sum')
        ts_trans['ts_vol'][:, :, ii] = out[0].T
        out1 = sx.grid2d(bin1_3D, bin2_3D, var_3D, 
                        xi=bin_edge1[:-1], yi=bin_edge2[:-1],
                        fn='count')
        ts_trans['ts_count'][:, :, ii] = out1[0].T
        
    for ii in np.arange(0, len(ts_trans.time)):
        bin1_3D = temperature_noflat[ii]
        bin2_3D = salinity_noflat[ii]
        
        # ts_area_mid
        var_3D=bin2_3D.copy()*0
        for i in range(var_3D.shape[0]):
            # Find the index of the last non-NaN value
            if open_or_close=='close':
                last_valid_index = None # np.where(~np.isnan(var_3D[i]))[0][-1] if np.any(~np.isnan(var_3D[i])) else None
            
            elif open_or_close=='open':
                last_valid_index = np.where(~np.isnan(var_3D[i]))[0][-1] if np.any(~np.isnan(var_3D[i])) else None
            # If a valid index was found, set all to 0 except the last non-NaN column
            if last_valid_index is not None:
                var_3D[i] = 0
                var_3D[i, last_valid_index] = dx*1000*dy
        

        out2 = sx.grid2d(bin1_3D.flatten(), bin2_3D.flatten(), var_3D.flatten(), 
                        xi=bin_edge1[:-1], yi=bin_edge2[:-1],
                        fn='sum')
        ts_trans['ts_area_boundary'][:, :, ii] = out2[0].T
    
    return ts_trans


def ts_trans_cdiff(ts_trans, dt, dens_limit=None):
    # Central difference
    dt = 2*dt
    ts_area_mid=ts_trans['ts_area_boundary'].isel(time=1).values
    
    ts_vol_plus1 = ts_trans['ts_vol'][: , : , 2:].copy() ##2
    ts_vol_minus1 = ts_trans['ts_vol'][: , : , :-2].copy() ## -2
    
    if dens_limit:
        ts_trans['lsqr_out'][:, :, ii] =out['lsqr_out']        
        S_g, T_g = np.meshgrid(ts_trans['S_bini'], ts_trans['T_bini'])
        
        dens_g = rho(S_g, T_g, 0) - 1000  ##rho imported from gsw

        dens_idx = (dens_g > dens_limit)

    for ii in np.arange(0, np.shape(ts_vol_plus1)[2], 1):

        ts_vol_start = ts_vol_minus1.isel(time=ii).data
        ts_vol_end = ts_vol_plus1.isel(time=ii).data
        
        if dens_limit:
            ts_vol_start[dens_idx] = 0
            ts_vol_end[dens_idx] = 0
            
        out = calc_tot_trans(ts_vol_start, ts_vol_end, ts_area_mid,  dt, ts_trans['S_bini'], ts_trans['T_bini'], ts_trans['S_binii'], ts_trans['T_binii'])

        ts_trans['G_t'][:, :, ii] = out['G_t']
        ts_trans['G_ti'][:, :, ii] = out['G_ti']
        ts_trans['G_s'][:, :, ii] = out['G_s']
        ts_trans['G_si'][:, :, ii] = out['G_si']
        ts_trans['dvol_G'][:, :, ii] = out['dvol'] * dt   #m2
        ts_trans['epsilon'][:, :, ii] = out['eps'] * dt
        ts_trans['lsq_relative_residual'][ii]=np.array(out['lsqr_out']['normr'])
        
    return ts_trans

# calc_tot_trans - compute dia-thermal and dia-haline transformations from volume change 
# Solve for the dia-thermal and dia-haline transformations responsible for 
# the volume change between time-steps. Build a series of matricies to solve 
# for the transformations e.g.
# A (coefficients) * x (transformations) = y (volume change)

def calc_tot_trans(ts_vol_start, ts_vol_end,ts_area_mid,  dt, S_bini, T_bini, S_binii, T_binii, iter_lim=500):

    m = len(T_bini)
    n = len(S_bini)
    
    vol = np.vstack( (ts_vol_start.flatten(order='F'), ts_vol_end.flatten(order='F') ))
    dvol = np.diff(vol, axis=0) / dt

    ########### Build y ############
    
    vol1_t1 = ts_vol_start[:-1, :].flatten(order='F')
    vol1_t2 = ts_vol_end[:-1, :].flatten(order='F')

    vol2_t1 = ts_vol_start[1:, :].flatten(order='F')
    vol2_t2 = ts_vol_end[1:, :].flatten(order='F')

    vol_t = np.nansum(np.vstack((vol1_t1, vol1_t2, vol2_t1, vol2_t2)) == 0, axis=0)

    zero_idx_t = np.where(vol_t != 4)[0]

    vol1_t1 = ts_vol_start[:, :-1].flatten(order='C')
    vol1_t2 = ts_vol_end[:, :-1].flatten(order='C')

    vol2_t1 = ts_vol_start[:, 1:].flatten(order='C')
    vol2_t2 = ts_vol_end[:, 1:].flatten(order='C')

    vol_s = np.nansum(np.vstack((vol1_t1, vol1_t2, vol2_t1, vol2_t2)) == 0, axis=0)
    vol_s = np.reshape(vol_s, (m, n - 1), order='C').flatten(order='F')

    zero_idx_s = np.where(vol_s != 4)[0]

    # dvol = (ts_vol_end - ts_vol_start) / dt

    # vol1_t1 = dvol[:-1, :].flatten(order='F')
    # vol2_t1 = dvol[1:, :].flatten(order='F')
    
    # vol_t = np.nansum(np.vstack((vol1_t1, vol2_t1)) == 0, axis=0)

    # zero_idx_t = np.where(vol_t != 2)[0]

    # vol1_t1 = dvol[:, :-1].flatten(order='C')
    # vol2_t1 = dvol[:, 1:].flatten(order='C')

    # vol_s = np.nansum(np.vstack((vol1_t1, vol2_t1)) == 0, axis=0)
    # vol_s = np.reshape(vol_s, (m, n - 1), order='C').flatten(order='F')

    # zero_idx_s = np.where(vol_s != 2)[0]

    y = dvol.flatten().copy()
    
##HERE####################### new compared to calc_tot_trans
    An = build_matrix(m, n, zero_idx_t, zero_idx_s)
    
    bndry_idx = np.where(ts_area_mid.flatten(order='F') > 0)[0]
    data = np.ones((len(bndry_idx)))
    A_eps = coo_matrix((data, (bndry_idx, bndry_idx)), shape=(m*n, m*n))
    
    An = hstack([An, A_eps])
###    

    yn = y
    
    ######### Solve for x using least squares minimisation #########
    x, istop, itn, normr = lsqr(An, yn, iter_lim=iter_lim)[:4]
    lsqr_out = {'istop': istop, 'itn': itn, 'normr': normr/np.linalg.norm(yn)}

    # # Split back in to isotherms and isohalines
    # xt = x[:(m - 1) * n].copy()
    # xs = x[(m - 1) * n:].copy()
    
# Split back in to isotherms and isohalines                     FEW CHANGES HERE TOO COMPARED t calc_tot_trans
    xt = x[:(m - 1) * n].copy()
    xs = x[(m - 1) * n : ((m - 1) * n) + (m * (n - 1))].copy()
    x_eps = x[((m - 1) * n) + (m * (n - 1)):]
###
    # Reshape in an array and interpolate for quiver plots
    G_t = np.reshape(xt, (m - 1, n), order='F')
    G_s = np.reshape(xs, (m, n - 1), order='F')

    f = interp2d(S_bini, T_binii, G_t, kind='linear', fill_value=np.nan)
    # f = RectBivariateSpline(S_bini, T_binii, G_t.T)
    # f = RegularGridInterpolator((S_bini, T_binii), G_t, method='linear', bounds_error=False, fill_value=np.nan)

    G_ti = f(S_bini, T_bini)


    f = interp2d(S_binii, T_bini, G_s, kind='linear', fill_value=np.nan)
    # f_raw = RectBivariateSpline(S_binii, T_bini, G_s.T)
    # # f = RegularGridInterpolator((S_binii, T_binii), G_s,  method='linear',  bounds_error=False, fill_value=np.nan)
    G_si = f(S_bini, T_bini)


    # Re-grid dvol - quicker than recalculating 
    dvol = np.reshape(dvol, (m, n), order='F')

### eps
    eps = np.reshape(x_eps, (m, n), order='F')
###

    tot_trans = {'G_t': G_t, 'G_s': G_s, 'G_ti': G_ti, 'G_si': G_si, 'dvol': dvol, 'eps': eps, 'lsqr_out': lsqr_out}   #m2s-1 ,,, dvol m2 s-1 eps m2s-1
    # tot_trans = {'G_t': G_t, 'G_s': G_s, 'G_ti': G_ti, 'G_si': G_si, 'dvol': dvol, 'lsqr_out': lsqr_out}

    return tot_trans



# build_matrix - build a sparse matrix representing the coeffcients for calc_tot_trans 
# It is important to remove the columns (isotherms) of the A matrix where there is no 
# chance of a flux. To do this check the bins adjacent to isotherm. If there was never 
# any volume in these bins, then there should be no flux across that isotherm, therefore
# the coefficient associated with that bin / iso-surface is set to zero.

def build_matrix(m, n, zero_idx_t, zero_idx_s):
    
    # Matrix indicies for isotherms
    c_idx_minus1 = np.zeros(((m - 1) * n))
    c_idx_plus1 = np.zeros(((m - 1) * n))
    r_idx_minus1 = np.zeros(((m - 1) * n))
    r_idx_plus1 = np.zeros(((m - 1) * n))

    c_idx = np.arange(0, ((m - 1) * n) + m-1, m-1)
    r_idx = np.arange(0, (m * n) + m, m)

    for idx in np.arange(0, len(c_idx)-1, 1):
        c_idx_minus1[c_idx[idx]:c_idx[idx+1]] = np.arange(c_idx[idx], c_idx[idx+1], 1)
        c_idx_plus1[c_idx[idx]:c_idx[idx+1]] = np.arange(c_idx[idx], c_idx[idx+1], 1)
        r_idx_minus1[c_idx[idx]:c_idx[idx+1]] = np.arange(r_idx[idx], r_idx[idx+1]-1)
        r_idx_plus1[c_idx[idx]:c_idx[idx+1]] = np.arange(r_idx[idx]+1, r_idx[idx+1])

    cols_t = np.hstack((c_idx_minus1.copy(), c_idx_plus1.copy())) 
    rows_t = np.hstack((r_idx_minus1.copy(), r_idx_plus1.copy())) 
    data_t = np.hstack((-np.ones(np.shape(c_idx_minus1)), np.ones(np.shape(c_idx_plus1)))) 

    # Remove colums where there should be no flux
    keep_idx_t = np.zeros(np.shape(cols_t), dtype=bool)

    for idx in np.arange(0, len(zero_idx_t), 1):
        keep_idx_t[cols_t == zero_idx_t[idx]] = True

    cols_t = cols_t[keep_idx_t]
    rows_t = rows_t[keep_idx_t]
    data_t = data_t[keep_idx_t]

    # Matrix indicies for isohalines 
    r_idx_minus1 = np.arange(0 , (m * n) - m, 1)
    r_idx_plus1 = np.arange(m , m * n, 1)

    c_idx_minus1 = np.arange(0, (n - 1) * m, 1)
    c_idx_plus1 = np.arange(0, (n - 1) * m, 1)

    cols_s = np.hstack((c_idx_minus1, c_idx_plus1)) 
    rows_s = np.hstack((r_idx_minus1, r_idx_plus1))
    data_s = np.hstack((-np.ones(np.shape(c_idx_minus1)), np.ones(np.shape(r_idx_plus1))))

    # Remove colums where there should be no flux
    keep_idx_s = np.zeros(np.shape(cols_s), dtype=bool)

    for idx in np.arange(0, len(zero_idx_s), 1):
        keep_idx_s[cols_s == zero_idx_s[idx]] = True

    cols_s = cols_s[keep_idx_s]
    rows_s = rows_s[keep_idx_s]
    data_s = data_s[keep_idx_s]
    
    cols = np.hstack((cols_t.copy(), cols_s.copy() + (m - 1) * n))
    rows = np.hstack((rows_t.copy(), rows_s.copy()))
    data = np.hstack((data_t.copy(), data_s.copy()))

    A = coo_matrix((data, (rows, cols)), shape=(m * n, (m-1)*n + (n-1)*m))    
    return A


def calc_osanp_F(ts_trans, dt):
    
    for ii in np.arange(0, len(ts_trans.time_mid), 1):

        out = calc_tot_trans_residual(ts_trans.dvol_F.isel(time_mid=ii).data, dt, ts_trans['S_bini'], ts_trans['T_bini'], ts_trans['S_binii'], ts_trans['T_binii'])

        ts_trans['F_t'][:, :, ii] = out['F_t']
        ts_trans['F_ti'][:, :, ii] = out['F_ti']
        ts_trans['F_s'][:, :, ii] = out['F_s']
        ts_trans['F_si'][:, :, ii] = out['F_si']

    return ts_trans

def calc_tot_trans_residual(dvol_res, dt, S_bini, T_bini, S_binii, T_binii, ts_area_mid=None, epsilon_bool=False, iter_lim=250):

    m = len(T_bini)
    n = len(S_bini)
    
    ########### Build y ############
    
#     vol1_t1 = ts_vol_start[:-1, :].flatten(order='F')
#     vol1_t2 = ts_vol_end[:-1, :].flatten(order='F')

#     vol2_t1 = ts_vol_start[1:, :].flatten(order='F')
#     vol2_t2 = ts_vol_end[1:, :].flatten(order='F')

#     vol_t = np.nansum(np.vstack((vol1_t1, vol1_t2, vol2_t1, vol2_t2)) == 0, axis=0)

#     zero_idx_t = np.where(vol_t != 4)[0]

#     zero_idx_t = None

    vol1_t1 = dvol_res[:-1, :].flatten(order='F')
    vol2_t1 = dvol_res[1:, :].flatten(order='F')
    
    vol_t = np.nansum(np.vstack((vol1_t1, vol2_t1)) == 0, axis=0)

    zero_idx_t = np.where(vol_t != 2)[0]
    

#     vol1_t1 = ts_vol_start[:, :-1].flatten(order='C')
#     vol1_t2 = ts_vol_end[:, :-1].flatten(order='C')

#     vol2_t1 = ts_vol_start[:, 1:].flatten(order='C')
#     vol2_t2 = ts_vol_end[:, 1:].flatten(order='C')

#     vol_s = np.nansum(np.vstack((vol1_t1, vol1_t2, vol2_t1, vol2_t2)) == 0, axis=0)
#     vol_s = np.reshape(vol_s, (m, n - 1), order='C').flatten(order='F')

#     zero_idx_s = np.where(vol_s != 4)[0]
    
#     zero_idx_s = None
    
    vol1_t1 = dvol_res[:, :-1].flatten(order='C')
    vol2_t1 = dvol_res[:, 1:].flatten(order='C')

    vol_s = np.nansum(np.vstack((vol1_t1, vol2_t1)) == 0, axis=0)
    vol_s = np.reshape(vol_s, (m, n - 1), order='C').flatten(order='F')

    zero_idx_s = np.where(vol_s != 2)[0]

    y = dvol_res.flatten(order='F').copy() / dt

    An = build_matrix(m, n, zero_idx_t, zero_idx_s)
    
    if epsilon_bool:
        bndry_idx = np.where(ts_area_mid.flatten(order='F') > 0)[0]
        data = np.ones((len(bndry_idx)))
        A_eps = coo_matrix((data, (bndry_idx, bndry_idx)), shape=(m*n, m*n))
    
        An = hstack([An, A_eps])
    
    yn = y
    
    ######### Solve for x using least squares minimisation #########
    x, istop, itn, normr = lsqr(An, yn, iter_lim=iter_lim)[:4]
    lsqr_out = {'istop': istop, 'itn': itn, 'normr': normr}

    # Split back in to isotherms and isohalines
    xt = x[:(m - 1) * n].copy()
    xs = x[(m - 1) * n : ((m - 1) * n) + (m * (n - 1))].copy()
    
    if epsilon_bool:
        x_eps = x[((m - 1) * n) + (m * (n - 1)):]
        eps = np.reshape(x_eps * dt, (m, n), order='F')
    
    # Reshape in an array and interpolate for quiver plots
    F_t = np.reshape(xt, (m - 1, n), order='F')
    F_s = np.reshape(xs, (m, n - 1), order='F')

    f = interp2d(S_bini, T_binii, F_t, kind='linear', fill_value=np.nan)
    F_ti = f(S_bini, T_bini) 

    f = interp2d(S_binii, T_bini, F_s, kind='linear', fill_value=np.nan)
    F_si = f(S_bini, T_bini) 
    
    if epsilon_bool:
        res_trans = {'F_t': F_t, 'F_s': F_s, 'F_ti': F_ti, 'F_si': F_si, 'dvol': dvol_res, 'eps': eps, 'lsqr_out': lsqr_out}
    else:
        res_trans = {'F_t': F_t, 'F_s': F_s, 'F_ti': F_ti, 'F_si': F_si, 'dvol': dvol_res, 'lsqr_out': lsqr_out}

    return res_trans


from gsw import z_from_p, p_from_z, distance, SA_from_SP, CT_from_t, rho, distance, cp_t_exact

# def calc_gr_area(longitude, latitude):
    
#     dlat = 0.5
#     dlon = 0.5

#     dx = np.repeat(distance(np.hstack( (np.zeros((len(latitude), 1)), np.ones((len(latitude), 1))*dlon )),
#                      np.hstack( (np.expand_dims(latitude, axis=1), np.expand_dims(latitude, axis=1) ) ) ),
#                    len(longitude), axis=1)

#     dy = np.repeat(distance(np.hstack( (np.expand_dims(longitude, axis=1), np.expand_dims(longitude, axis=1)) ), 
#                      np.hstack( (np.zeros((len(longitude), 1)), np.ones((len(longitude), 1))*dlat) ) ),
#                   len(latitude), axis=1)
#     dy = np.transpose(dy)

#     gr_area = dx * dy
    
#     return gr_area

# Solve for the volume change due to air-sea fluxes.
# Just use straight forward matrix multiplication:
# A (coefficients) * x (transformations) = y (volume change)
# Use this if you want a residual volume change,
# i.e. total volume change minus air-sea flux volume change.

def calc_surf_trans(ts_trans, qnet, fnet, sst, sss, longitude, latitude, use_gsw=True):
    
    gr_area = 30000*30000#alc_gr_area(longitude, latitude)

    S_gridt, T_gridt = np.meshgrid(ts_trans['S_bini'], ts_trans['T_bin'])
    S_grids, T_grids = np.meshgrid(ts_trans['S_bin'], ts_trans['T_bini'])

    cp = cp_t_exact(S_gridt, T_gridt, 0)

    if use_gsw:
        dens = rho(S_gridt, T_gridt, 0)
    else:
        dens = sw.dens0(S_gridt, T_gridt)     
        
    T_const = np.expand_dims(cp * dens * ts_trans.dT.data, axis=2)
    S_const = np.expand_dims(S_grids / ts_trans.dS.data, axis=2)

    for ii in np.arange(0, len(ts_trans.time_mid), 1):

        # Dia-thermal transformations 
        bin_edge1 = ts_trans['T_binj']
        bin_edge2 = ts_trans['S_bin']

        
#         if use_gsw:
#             SA = SA_from_SP(sss.isel(time=ii), 0, longitude, latitude)
#             CT = CT_from_t(SA, sst.isel(time=ii), 0)

#             bin1 = CT.compute().data.flatten()
#             bin2 = SA.compute().data.flatten()
#         else:
        bin1 = sst[ii]#flatten()
        bin2 = sss[ii]#flatten()
            
        var = (sst[ii]*0)+(qnet.isel(time_mid=ii).compute().data * gr_area).flatten()

        # bin1[np.isnan(var)] = np.nan
        # bin2[np.isnan(var)] = np.nan
        out = sx.grid2d(bin1, bin2, var, 
                                xi=bin_edge1[:-1], yi=bin_edge2[:-1],
                                fn='sum')
        # out = binned_statistic_2d(bin1, bin2, var, statistic='sum', bins=(bin_edge1, bin_edge2))

        ts_trans['E_t'][:, :, ii] = out[0].T.copy()

        # Dia-haline transformations 
        bin_edge1 = ts_trans['T_bin']
        bin_edge2 = ts_trans['S_binj']

        # var = (fnet.isel(time_mid=ii).compute().data * gr_area).flatten()
        var = (sst[ii]*0)+(fnet.isel(time_mid=ii).compute().data * gr_area).flatten()

        out = sx.grid2d(bin1, bin2, var, 
                                xi=bin_edge1[:-1], yi=bin_edge2[:-1],
                                fn='sum')
        
        ts_trans['E_s'][:, :, ii] = out[0].T.copy()

    ts_trans['E_t'] = ts_trans['E_t'] / T_const
    ts_trans['E_s'] = ts_trans['E_s'] * S_const

    ts_trans['E_ti'] = ts_trans['E_t'].interp(T_bin=ts_trans.T_bini)
    ts_trans['E_si'] = ts_trans['E_s'].interp(S_bin=ts_trans.S_bini)
    
    ts_trans = dvol_all(ts_trans, (30*24*60*60))
    
    return ts_trans

def dvol_all(ts_trans, dt):

    for ii in np.arange(0, len(ts_trans['time_mid']), 1):

        ts_trans['dvol_E'][:, :, ii] = calc_E_trans_dvol(ts_trans['E_t'][:, :, ii].data,  ts_trans['E_s'][:, :, ii].data, 
                                             ts_trans['T_bini'], ts_trans['S_bini'], dt)
    return ts_trans

def calc_E_trans_dvol(E_t, E_s, T_bini, S_bini, dt):

    m = len(T_bini)
    n = len(S_bini)

    ########### Build A and x  ############

    xt_nonsparse = E_t.flatten(order='F')
    xs_nonsparse = E_s.flatten(order='F')

    dat_idx = np.where(xt_nonsparse!=0)[0]
    xt = coo_matrix((xt_nonsparse[dat_idx], (dat_idx, np.zeros(dat_idx.shape))), shape=((m+1) * n, 1))

    dat_idx = np.where(xs_nonsparse!=0)[0]
    xs = coo_matrix((xs_nonsparse[dat_idx], (dat_idx, np.zeros(dat_idx.shape))), shape=(m * (n+1), 1))

    x = vstack([xt, xs])    

    zero_idx_t = None
    zero_idx_s = None

    A = build_matrix_E_trans(m, n, zero_idx_t, zero_idx_s)
    dvol = np.reshape(A.dot(x).toarray() * dt, (m, n), order='F')
    return dvol


def build_matrix_E_trans(m, n, zero_idx_t, zero_idx_s):
    
    # Matrix indicies for isotherms
    c_idx_minus1 = np.zeros(((m + 1) * n))
    c_idx_plus1 = np.zeros(((m + 1) * n))
    r_idx_minus1 = np.zeros(((m + 1) * n))
    r_idx_plus1 = np.zeros(((m + 1) * n))

    c_idx = np.arange(0, ((m + 1) * n) + m+1, m+1)
    r_idx = np.arange(0, (m * n) + m, m)

    for idx in np.arange(0, len(c_idx)-1, 1):
        c_idx_minus1[c_idx[idx]:c_idx[idx+1]-1] = np.arange(c_idx[idx]+1, c_idx[idx+1], 1)
        c_idx_plus1[c_idx[idx]:c_idx[idx+1]-1] = np.arange(c_idx[idx], c_idx[idx+1]-1, 1)
        r_idx_minus1[c_idx[idx]:c_idx[idx+1]-1] = np.arange(r_idx[idx], r_idx[idx+1])
        r_idx_plus1[c_idx[idx]:c_idx[idx+1]-1] = np.arange(r_idx[idx], r_idx[idx+1])

    cols_t = np.hstack((c_idx_minus1.copy(), c_idx_plus1.copy())) 
    rows_t = np.hstack((r_idx_minus1.copy(), r_idx_plus1.copy())) 
    data_t = np.hstack((-np.ones(np.shape(c_idx_minus1)), np.ones(np.shape(c_idx_plus1)))) 

    if zero_idx_t:
        # Remove colums where there should be no flux
        keep_idx_t = np.zeros(np.shape(cols_t), dtype=bool)

        for idx in np.arange(0, len(zero_idx_t), 1):
            keep_idx_t[cols_t == zero_idx_t[idx]] = True

        cols_t = cols_t[keep_idx_t]
        rows_t = rows_t[keep_idx_t]
        data_t = data_t[keep_idx_t]

    # Matrix indicies for isohalines 
    r_idx_minus1 = np.arange(0 , (m * n), 1)
    r_idx_plus1 = np.arange(0 , (m * n), 1)

    c_idx_minus1 = np.arange(m, (n + 1) * m, 1)
    c_idx_plus1 = np.arange(0, ((n + 1) * m) - m, 1)

    cols_s = np.hstack((c_idx_minus1, c_idx_plus1)) 
    rows_s = np.hstack((r_idx_minus1, r_idx_plus1))
    data_s = np.hstack((-np.ones(np.shape(c_idx_minus1)), np.ones(np.shape(r_idx_plus1))))

    if zero_idx_s:
        # Remove colums where there should be no flux
        keep_idx_s = np.zeros(np.shape(cols_s), dtype=bool)

        for idx in np.arange(0, len(zero_idx_s), 1):
            keep_idx_s[cols_s == zero_idx_s[idx]] = True

        cols_s = cols_s[keep_idx_s]
        rows_s = rows_s[keep_idx_s]
        data_s = data_s[keep_idx_s]

    cols = np.hstack((cols_t.copy(), cols_s.copy() + ((m + 1) * n)))
    rows = np.hstack((rows_t.copy(), rows_s.copy()))
    data = np.hstack((data_t.copy(), data_s.copy()))

    A = coo_matrix((data, (rows, cols)), shape=(m * n, ((m+1)*n) + (m*(n+1))))  
    return A
