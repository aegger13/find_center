import numpy as np
import itertools
from skimage import feature
from numba import jit
from scipy import sparse
from scipy import optimize
from skimage.draw import circle_perimeter
from skimage.measure import (CircleModel, ransac)
from scipy.signal import argrelextrema
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from time import time

# Canny
CAN_SIG = 4  # Canny sigma threshold
LO_THRESH = 0.92  # canny lower threshold for edge connection
HI_THRESH = 0.98  # canny upper threshold for edge connection

# Search Params
OVERFILL = 1.5 # pixels
C_BIN = 100  # bins for center coordinates in hough space
R_BIN = 280  # Bins for radius in hough space
R_RANGE = [1, 1401]  # No clue how default came to be
WT = 1. # weighting factor for adding point to hough space
PREC = 1
RED_FACTOR = 5.
NORM = False

def find_edges(image, mask, sigma=CAN_SIG, hi_thresh=HI_THRESH, low_thresh=LO_THRESH, plot=False):
    """Run the canny edge detection"""
    edges = feature.canny(image, mask=mask.astype(bool), sigma=CAN_SIG , low_threshold=LO_THRESH, \
                      high_threshold=HI_THRESH, use_quantiles=True)
    sparse_edges = sparse.coo_matrix(edges)
    
    if plot:
        plt.figure()
        plt.imshow(edges)
        plt.show()

    return sparse_edges

@jit(cache=True, nopython=True)
def produce_hough_array(radii, center_x, center_y, ar_hough, zip_obj):
    """iterate through edges and add to hough space"""
    delta_r = radii[1] - radii[0]
    r_low = radii[0] ** 2
    r_hi = radii[-1] ** 2
    for row, col, data in zip_obj:
        for ix, cx in enumerate(center_x):
            dx = (row - cx) ** 2
            if dx < r_low or  dx > r_hi:
                # skip iteration if outside bounds
                continue
            for iy, cy in enumerate(center_y):
                dy = (col - cy) ** 2
                r = (dx + dy) ** 0.5
                ir = int((r - radii[0]) / delta_r)
                if ir >= 0 and ir < radii.shape[0]:
                    ar_hough[ir, ix, iy] += data

    return ar_hough

def run_hough(ar_hough, sparse_edges, x_range, y_range, radii, center_x, center_y):
    """specify data outside of jit compiled function, then produce hough array"""
    # create zip object outside of jit compiled function
    zip_obj = zip(sparse_edges.row, sparse_edges.col, sparse_edges.data)
    # add edges to hough space (r, x, y)
    ar_hough = produce_hough_array(radii, center_x, center_y, ar_hough, zip_obj)

    return ar_hough

def max_from_hough(ar_hough, radii, center_x, center_y):
    """Get maximum value from hough space for r, x, and y"""
    maxdim_r = [ar_hough[i,:,:].max() for i in range(ar_hough.shape[0])]
    maxdim_x = [ar_hough[:,i,:].max() for i in range(ar_hough.shape[1])]
    maxdim_y = [ar_hough[:,:,i].max() for i in range(ar_hough.shape[2])]

    r_max = radii[np.array(maxdim_r).argmax()]
    x_max = center_x[np.array(maxdim_x).argmax()]
    y_max = center_y[np.array(maxdim_y).argmax()]

    print(' get max ', center_x, center_y)

    return r_max, x_max, y_max, maxdim_r

def iterate_center(sparse_edges, overfill=OVERFILL, r_range=R_RANGE, r_bin=R_BIN, c_bin=C_BIN, prec=PREC, red_factor=RED_FACTOR):
    """Iterate through center finding until we are within defined precision"""
    # Image ranges based on size and overfill factor in pixels
    x_range = [sparse_edges.shape[0] * (1. - overfill), sparse_edges.shape[0] * overfill]
    y_range = [sparse_edges.shape[1] * (1. - overfill), sparse_edges.shape[1] * overfill]
    radii = np.arange(r_range[0], r_range[1], (r_range[1] - r_range[0]) / r_bin)
    # Center guesses based on center bins and image range
    center_x = np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / c_bin)
    center_y = np.arange(y_range[0], y_range[1], (y_range[1] - y_range[0]) / c_bin)
    ar_hough = np.zeros([radii.shape[0], center_x.shape[0], center_y.shape[0]])
    ar_hough = run_hough(ar_hough, sparse_edges, x_range, y_range, radii, center_x, center_y)
    r_max, x_max, y_max, maxdim_r = max_from_hough(ar_hough, radii, center_x, center_y)

    while (center_x[1] - center_x[0]) > prec:
        r_size_x = (x_range[1] - x_range[0]) / red_factor
        r_size_y = (y_range[1] - y_range[0]) / red_factor
        x_range = [x_max - r_size_x * 0.5, x_max + r_size_x * 0.5]
        y_range = [y_max - r_size_y * 0.5, y_max + r_size_y * 0.5]
        center_x = np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / c_bin)
        center_y = np.arange(y_range[0], y_range[1], (y_range[1] - y_range[0]) / c_bin)
        arHough = np.zeros([radii.shape[0], center_x.shape[0], center_y.shape[0]])
        ar_hough = run_hough(ar_hough, sparse_edges, x_range, y_range, radii, center_x, center_y)
        r_max, x_max, y_max, maxdim_r = max_from_hough(ar_hough, radii, center_x, center_y)

    temp = x_max
    x_max = y_max
    y_max = temp
    print('result ', r_max, x_max, y_max)
    return r_max, x_max, y_max, radii, maxdim_r

def find_r_max(radii, maxdim_r, norm=NORM, min_dr=-1):
    """Get the r range for search"""
    if norm:
        r_max = argrelextrema(np.array(maxdim_r) / radii, np.greater)
    else:
        r_max = argrelextrema(np.array(maxdim_r), np.greater)

    max_radii = radii[r_max[0]]

    if max_radii.shape[0] == 0:
        print('could not a maxima in radii')
        return []

    res_at_radii = np.array(maxdim_r)[r_max[0]]
    res_at_radii, max_radii = (list(x) for x in zip(*sorted(zip(res_at_radii, max_radii))))

    nrad = []
    cur_min_dr = 1e6
    for rad in reversed(max_radii):
        for irrad in nrad:
            diff_val = np.fabs(rad - irrad)
            if diff_val < cur_min_dr:
                cur_min_dr = diff_val
        if cur_min_dr > min_dr:
            nrad.append(rad)

    return nrad

def find_points_in_ring(sparse_edges, center, r_max, n_max=10, delta_r=5, min_points=40, min_samples=10, \
    res_thresh=2, min_frac=0.45):
    """Find the number of edges inside proposed ring, if exceeds threshold, use ransac
    to fit the circle
    """
    tree = cKDTree(np.array([sparse_edges.col, sparse_edges.row]).T)
    ring_info = []
    for ir, r in enumerate(r_max):
        if len(ring_info) >= n_max:
            break

        cur_ring_info = {}
        r_outer = tree.query_ball_point(center, r + delta_r)
        r_inner = tree.query_ball_point(center, r - delta_r)
        in_ring = set(r_outer).symmetric_difference(set(r_inner))
        cur_ring_info['r_input'] = r
        cur_ring_info['points_in_circle'] = list(in_ring)

        if len(list(in_ring)) < min_points:
            continue

        model, inliers = ransac(np.array([sparse_edges.row[list(in_ring)], sparse_edges.col[list(in_ring)]]).T, \
            CircleModel, min_samples=min_samples, residual_threshold=res_thresh, max_trials=1000)
        
        cur_ring_info['in_frac'] = inliers.astype(int).sum() / float(len(list(in_ring)))
        cur_ring_info['inliers'] = inliers
        cur_ring_info['ransac_result'] = model

        if cur_ring_info['in_frac'] > min_frac and inliers.astype(int).sum() >= min_points:
            ring_info.append(cur_ring_info)

    return ring_info

def fit_circle(x, y, yerr=None, guess=None):
    """
    fit a single circle. Transform input to lists to that fitCircles can be used.
    """  
    x = [x]
    y = [y]
    #largest differences/2/2
    rGuess = (np.nanmax(x) - np.nanmin(x) + np.nanmax(y) - np.nanmin(y)) / 4.
    r = [rGuess]
    fit_res = fit_circles(x, y, r, yerr=None, guess=None)
    #have only one circle.
    fit_res['R'] = fit_res['R'][0]
    fit_res['residu'] = fit_res['residu'][0]
    return fit_res

def fit_circles(x, y, r, yerr=None, guess=None):
  """
  simultanous least squares fitting of multiple concentric rings.
  """  
  def calc_r(x, y, par):
      xc, yc = par
      return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

  def f(par, x, y):
      ri = calc_r(x, y, par)
      return ri - ri.mean()

  def f_global(par, mat_r, mat_x, mat_y):
      err = []
      for r, x, y in zip(mat_r, mat_x, mat_y):
          err_local = f(par, x, y)
          err = np.concatenate((err, err_local))          
      return err

  if guess is None or len(guess) != 2:
      x_m = np.mean(x[0])
      y_m = np.mean(y[0])
  else:
      x_m = guess[0]
      y_m = guess[1]
      
  center_estimate = x_m, y_m
  fit_res = {}  
  if yerr is not None:      
      center, C, info, msg, success  = optimize.leastsq(f_global, \
          center_estimate, args=(r, x, y), full_output=True)
      fit_res['C'] = C
      fit_res['info'] = info
      fit_res['msg'] = msg
      fit_res['success'] = success
  else:
      center, ier = optimize.leastsq(f_global, center_estimate, args=(r, x, y))
      fit_res['ier'] = ier
  xc, yc = center
  fit_res['xCen'] = xc
  fit_res['yCen'] = yc
  rs=[]
  resids=[]
  for thisr, thisx, thisy in zip(r,x,y):
      ri     = calc_r(thisx, thisy, center)
      rs.append(ri.mean())
      resids.append(np.sum((ri - ri.mean()) ** 2))
  fit_res['residu'] = resids
  fit_res['R']      = rs

  return fit_res
