# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:19:07 2026

@author: ed
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import cv2


##########################  CONFIGURATION & CONSTANTS   ################################################

#FILE_PATH = r"C:\Users\ed\OneDrive - University of Bristol\Documents\ASC\Pinhole regions2" # Old Data
FILE_PATH = r"C:\Users\ed\OneDrive - University of Bristol\Documents\ASC\Regions ACCURATE" # New Data

# Data Coordinates
WIDTH, HEIGHT = 2080, 1520
X0_DATA, Y0_DATA = 1040, 760 # Image centre 
X0_ZENITH, Y0_ZENITH = 1036.221, 691.03862 # Data centre (dome zenith)
X_OFFSET, Y_OFFSET = (X0_DATA - X0_ZENITH), (Y0_DATA - Y0_ZENITH) # Image/ data offset, 

# Grid for Modeling
altitudes = np.arange(0, 90, 10)
azimuths = np.linspace(0, 360, 36, endpoint=False)

# Initial Model Parameters
f_initial = 368 # Initial f function
x0_dome = X0_ZENITH # Dome Center x
y0_dome = Y0_ZENITH # Dome Center y
alpha_rot = np.radians(0) # Lens Rotation


#########################  DATA LOADING ################################################################

x_raw_list, y_raw_list, = [], [],
with open(FILE_PATH, "r") as file:
    for line in file:
        match = re.search(r'circle\(([^,]+),([^,]+),([^)]+)\)', line) # Pulling the x and y coordinates
        if match:
            x_raw_list.append(float(match.group(1))) # Taking the coordinates as lists
            y_raw_list.append(float(match.group(2)))

x_raw = np.array(x_raw_list) # Converting them to arrays
y_raw = np.array(y_raw_list)
# Calculate the radial distance from the dome center
r_raw = np.sqrt((x_raw - x0_dome)**2 + (y_raw - y0_dome)**2)

print(f" Loaded {len(x_raw)} pinholes.")

plt.figure(figsize=(6, 6))
plt.scatter(x_raw, y_raw, s=30, facecolors='none', edgecolors='black', alpha=0.6)
plt.title("Pinhole Dome Data")
plt.gca().set_aspect('equal')
plt.grid(True, alpha=0.3)
plt.show()

###################### ERROR LOADING ####################################################

ERROR_FILEPATH = r"C:\Users\ed\OneDrive - University of Bristol\Documents\ASC\Region Error"

altitude_stds = {}
altitudes_err = [0, 10, 20, 30, 40, 50, 60, 70, 80]

for alt in altitudes_err:
    std_alt = []
    # Different number of pinholes for zenith
    if alt == 0:
        pinhole_range = range(1, 2)  # Ensuring the loop works with alt 0, where there is only 1 sample 
    else:
        pinhole_range = range(1, 7)   
    for n in pinhole_range:
        
        filepath = f"{ERROR_FILEPATH}\\alt_{alt}_pinhole{n}"
        x_err = []
        y_err = []
        
        with open(filepath, "r") as file:
            for line in file:
                match = re.search(r'circle\(([^,]+),([^,]+),([^)]+)\)', line) # Interpreting DS9 region file
                if match:
                    x_err.append(float(match.group(1)))
                    y_err.append(float(match.group(2)))
        
        x = np.array(x_err)
        y = np.array(y_err)
        
        std_x = np.std(x)
        std_y = np.std(y)
        std_total = np.sqrt(std_x**2 + std_y**2) # Finding radial uncertainty with x and y std
        
        std_alt.append(std_total)
    
        altitude_stds[alt] = std_alt

alts = []
mean_stds = []

print("\nCentroid Uncertainty with Altitude")
print(f"{'Altitude':<10} | {'Mean Std (px)':<15} | {'Samples':<10}")
print("-" * 40)

for alt in sorted(altitude_stds.keys()):
    vals = altitude_stds[alt]
    mean_std = np.mean(vals) 
    
    alts.append(alt)
    mean_stds.append(mean_std)
    
    print(f"{alt:<10} | {mean_std:<15.4f} | {len(vals):<10}")

plt.figure(figsize=(8, 5))
plt.plot(alts, mean_stds, marker='o')

plt.xlabel("Altitude (degrees)")
plt.ylabel("Mean Centroid Std (pixels)")
plt.title("Centroiding Uncertainty vs Altitude")

plt.grid(True, alpha=0.3)
plt.show()


##################### THE PROJECTION MODEL #####################################################


def project_to_pixel(alt_deg, azi_deg, f, x0, y0, alpha): 
    """Building ideal f-theta projection function including rotation"""
    theta = np.radians(alt_deg) # Converting angles to radians
    phi = np.radians(azi_deg)

    r = f * theta # Taking a standard f theta projection
    x_plan = r * np.cos(phi) # Converting these polar coordinates to a cartesian plane
    y_plan = r * np.sin(phi)

    xr = x_plan * np.cos(alpha) - y_plan * np.sin(alpha) # Applying lens rotation
    yr = x_plan * np.sin(alpha) + y_plan * np.cos(alpha)

    x_pix = xr + x0 # Mapping and returning image pixel coordinates
    y_pix = y0 - yr
    return x_pix, y_pix

def get_model_coords(f_val, x0_val, y0_val, alpha_val):
    """Generating a full grid and projecting it onto the data coordinate frame."""
    xs_mod, ys_mod = [], [] # Building up lists of x and y positions for all model points
    for alt in altitudes: # Looping over altitude and azimuth
        if alt == 0: # Ensuring only 1 point at zenith
            px, py = project_to_pixel(0, 0, f_val, x0_val, y0_val, alpha_val)
            xs_mod.append(px - X_OFFSET + (X0_DATA - x0_val))
            ys_mod.append(py - Y_OFFSET + (Y0_DATA - y0_val))
            continue
        for azi in azimuths:
            # Take each alt azi direction and project it using previous function
            px, py = project_to_pixel(alt, azi, f_val, x0_val, y0_val, alpha_val) 

            xs_mod.append(px - X_OFFSET + (X0_DATA - x0_val)) # Shift model coordinates to the data frame
            ys_mod.append(py - Y_OFFSET + (Y0_DATA - y0_val))
    return np.column_stack((xs_mod, ys_mod)) # Return as numpy array directly comparable to raw data.

# RMS check for current parameters
model_pts = get_model_coords(f_initial, x0_dome, y0_dome, alpha_rot)

tree = cKDTree(model_pts)
dist, _ = tree.query(np.column_stack((x_raw, y_raw)))

print(f"Initial RMS error: {np.sqrt(np.mean(dist**2)):.4f} px")





####################### FOCAL LENGTH OPTIMIZATION ######################################################


def objective_f(f_test):  
    """Iterates over different values of f and returns the mean squared error, and then
    gives the value of f with the lowest MSE."""
    # Generates model points with a certain trial focal length
    model_pts = get_model_coords(f_test, x0_dome, y0_dome, alpha_rot) 
    
    # Builds a nearest-neighbour algorithm from model points
    tree = cKDTree(model_pts)       
    # Finds distance from data point to closest model point                                     
    dist, _ = tree.query(np.column_stack((x_raw, y_raw))) 
         
    return np.mean(dist**2) # Returns mean squared distance as the error                                         

# Searches for the f that minimises the error within these bounds
opt_f = minimize_scalar(objective_f, bounds=(100, 1100), method='bounded')  
f_best = opt_f.x # Extracts the optimal focal length 
print(f"Best Focal Length: {f_best:.4f} px")                             
print(f"New Baseline RMS: {np.sqrt(opt_f.fun):.4f} px")

f_values = np.linspace(100, 1100, 100)  # Looping focal length and finding error
errors = []

for f in f_values:
    errors.append(objective_f(f))

errors = np.array(errors) # Plotting Error with focal length
plt.figure(figsize=(8, 5))
plt.plot(f_values, errors)
plt.axvline(100, linestyle='--', label='Lower bound (100)')
plt.axvline(1100, linestyle='--', label='Upper bound (1100)')
plt.axvline(f_best, linestyle='-', label=f'Best f ≈ {f_best:.1f}')

plt.xlabel("Focal Length f (pixels)")
plt.ylabel("Mean Squared Error")
plt.title("MS Error vs Focal Length")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()





########################## ROTATION OPTIMIZATION ############################################################


def objective_alpha(alpha_test):
    """Iterates over different values of alpha (radians) and returns the mean squared error, and then
    gives the value of alpha with the lowest MSE."""
    # Generates model points with the optimal f value, and an initial alpha value.
    model_pts = get_model_coords(f_best, x0_dome, y0_dome, alpha_test)
    
    # Creates a nearest neighbour lookup for the model pts
    tree = cKDTree(model_pts)
    # Finds distance from data point to closest model point 
    dist, _ = tree.query(np.column_stack((x_raw, y_raw)))
    
    return np.mean(dist**2) # Returns mean squared distance as the error   


# Searches for the alpha value that minimises the error within these bounds
opt_alpha = minimize_scalar(objective_alpha, bounds=(-np.pi/18, np.pi/18), method='bounded')
best_alpha_rad = opt_alpha.x
best_alpha_deg = np.degrees(best_alpha_rad)

print(f"Best Alpha: {best_alpha_rad:.6f} rad ({best_alpha_deg:.4f} degrees)")
print(f"New Baseline RMS: {np.sqrt(opt_alpha.fun):.4f} px")

alpha_values = np.linspace(-np.pi/18, np.pi/18, 720) # Looping alpha and finding error
errors = []

for alpha in alpha_values:
    errors.append(objective_alpha(alpha))

errors = np.array(errors) # Plotting Error with alpha
plt.figure(figsize=(8, 5))
plt.plot(alpha_values, errors)
plt.axvline(best_alpha_rad, linestyle='-', label=f'Best alpha ≈ {best_alpha_rad:.3f} rad')

plt.xlabel("Alpha (radians)")
plt.ylabel("Mean Squared Error")
plt.title("Mean Squared Error vs Lens Rotation Alpha")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()



########################## TILT OPTIMIZATION #############################################


def get_model_coords_tilt(f_val, x0_val, y0_val, alpha_val, tilt_x=0.0, tilt_y=0.0):
    """Generating model coords for given pitch and roll"""
    xs_mod, ys_mod = [], []

    for alt in altitudes:
        for azi in azimuths:
            theta = np.radians(alt) # Converting alt and azi into 3D coordinates
            phi = np.radians(azi)
            vx = np.sin(theta) * np.cos(phi)
            vy = np.sin(theta) * np.sin(phi)
            vz = np.cos(theta)

            # Applying tilt_x (pitch) rotation around x-axis
            vy_new = vy * np.cos(tilt_x) - vz * np.sin(tilt_x)
            vz_new = vy * np.sin(tilt_x) + vz * np.cos(tilt_x)
            vy, vz = vy_new, vz_new

            # Applingy tilt_y (roll) rotation around y-axis
            vx_new = vx * np.cos(tilt_y) + vz * np.sin(tilt_y)
            vz_new = -vx * np.sin(tilt_y) + vz * np.cos(tilt_y)
            vx, vz = vx_new, vz_new

            # Converting back to alt and azi 
            alt_new = np.degrees(np.arccos(vz))
            azi_new = np.degrees(np.arctan2(vy, vx))
            
            # Projecting these new tilted alt and azi coordinates onto a plane. 
            px, py = project_to_pixel(alt_new, azi_new, f_val, x0_val, y0_val, alpha_val)

            # Mapping back to data frame.
            xs_mod.append(px - X_OFFSET + (X0_DATA - x0_val))
            ys_mod.append(py - Y_OFFSET + (Y0_DATA - y0_val))

    return np.column_stack((xs_mod, ys_mod))

def objective_tilt(tilt_comb):
    """Returns mean squared error between tilted model point and raw data"""
    # Generates model points with the optimal f and alpha, and an initial tilt
    model_pts = get_model_coords_tilt(f_best, x0_dome, y0_dome, best_alpha_rad,
                                         tilt_x=tilt_comb[0], tilt_y=tilt_comb[1])
    tree = cKDTree(model_pts) # Finding MSE between nearest neighbours
    dist, _ = tree.query(np.column_stack((x_raw, y_raw)))
    return np.mean(dist**2) # MSE returned

# Optimizer minimizes MSE to find optimal tilt between -0.3 and 0.3 rad, starting at 0. I use the BFGS algorithm with a 
# limiter to save memory, and ensure it is bounded.
best_tilt = minimize(objective_tilt, x0=[0.0, 0.0], bounds= [(-0.3, 0.3), (-0.3, 0.3)], method='L-BFGS-B')

best_tilt_x, best_tilt_y = best_tilt.x
print(f"Best Tilt X: {best_tilt_x:.6f} rad ({np.degrees(best_tilt_x):.3f} deg)")
print(f"Best Tilt Y: {best_tilt_y:.6f} rad ({np.degrees(best_tilt_y):.3f} deg)")
print(f"New Baseline RMS with tilt: {np.sqrt(best_tilt.fun):.4f} px")


# Sweep for Tilt X (pitch, holding tilt_y = 0)
tilt_x_vals = np.linspace(-0.3, 0.3, 100)  # radians (~±5.7 deg)
errors_x = []

for tx in tilt_x_vals:
    errors_x.append(objective_tilt([tx, 0.0]))

errors_x = np.array(errors_x)

# Sweep for Tilt Y (roll, holding tilt_x = 0)
tilt_y_vals = np.linspace(-0.3, 0.3, 100)
errors_y = []

for ty in tilt_y_vals:
    errors_y.append(objective_tilt([0.0, ty]))

errors_y = np.array(errors_y)

# Plotting error with tilt
plt.figure(figsize=(10, 5))

plt.plot(np.degrees(tilt_x_vals), errors_x, label='Tilt X Sweep', color='blue')
plt.plot(np.degrees(tilt_y_vals), errors_y, label='Tilt Y Sweep', color='green')
plt.axvline(np.degrees(best_tilt_x), linestyle='--', color='blue', label=f'Best Tilt X ≈ {np.degrees(best_tilt_x):.2f}°')
plt.axvline(np.degrees(best_tilt_y), linestyle='--', color='green', label=f'Best Tilt Y ≈ {np.degrees(best_tilt_y):.2f}°')

plt.xlabel("Tilt Angle (degrees)")
plt.ylabel("Mean Squared Error")
plt.title("MS Error vs Tilt Angle")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



######################### MATCHING PAIRS #################################################################

# Generating the final optimised physical model

# Tilt included model
#model_final = get_model_coords_tilt(f_best, x0_dome, y0_dome, best_alpha_rad, tilt_x=best_tilt_x, tilt_y=best_tilt_y)
 
# Tilt ommited model                      
model_final = get_model_coords(f_best, x0_dome, y0_dome, best_alpha_rad)

tree = cKDTree(model_final) # Building pair matching
distances, indices = tree.query(np.column_stack((x_raw, y_raw)))

# Extracting matched pairs and computing residuals
model_match = model_final[indices]
res_x = x_raw - model_match[:, 0]
res_y = y_raw - model_match[:, 1]
rms_phys = np.sqrt(np.mean(res_x**2 + res_y**2))




####################### PHYSICAL MODEL PLOTS #######################################################

def get_matched_model(model_pts):
    """Matches raw data and model pairs"""
    tree = cKDTree(model_pts)
    _, indices = tree.query(np.column_stack((x_raw, y_raw)))
    return model_pts[indices]

### Initial Model vs Data #####

model_pre_f = get_model_coords(f_initial, x0_dome, y0_dome, alpha_rot)
model_pre_f_match = get_matched_model(model_pre_f)

plt.figure(figsize=(6, 6))
plt.scatter(x_raw, y_raw, s=30, facecolors='none', edgecolors='black', alpha=0.6, label='Data')
plt.scatter(model_pre_f_match[:, 0], model_pre_f_match[:, 1], s=15, marker='x', label='Model')

plt.title("Pre-Optimisation (Initial f)")
plt.gca().set_aspect('equal')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

### Optimised f vs data #####

model_post_f = get_model_coords(f_best, x0_dome, y0_dome, alpha_rot)
model_post_f_match = get_matched_model(model_post_f)

plt.figure(figsize=(6, 6))
plt.scatter(x_raw, y_raw, s=30, facecolors='none', edgecolors='black', alpha=0.6, label='Data')
plt.scatter(model_post_f_match[:, 0], model_post_f_match[:, 1], s=15, marker='x', label='Model')

plt.title("Post Focal Length Optimisation")
plt.gca().set_aspect('equal')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


#### Optimised alpha vs data #####

model_post_rot = get_model_coords(f_best, x0_dome, y0_dome, best_alpha_rad)
model_post_rot_match = get_matched_model(model_post_rot)

plt.figure(figsize=(6, 6))
plt.scatter(x_raw, y_raw, s=30, facecolors='none', edgecolors='black', alpha=0.6, label='Data')
plt.scatter(model_post_rot_match[:, 0], model_post_rot_match[:, 1], s=15, marker='x', label='Model')

plt.title("Post Rotation Optimisation")
plt.gca().set_aspect('equal')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#### Optimised tilt vs data #####

model_post_tilt = get_model_coords_tilt(
    f_best, x0_dome, y0_dome, best_alpha_rad,
    tilt_x=best_tilt_x, tilt_y=best_tilt_y)

model_post_tilt_match = get_matched_model(model_post_tilt)

plt.figure(figsize=(6, 6))
plt.scatter(x_raw, y_raw, s=30, facecolors='none', edgecolors='black', alpha=0.6, label='Data')
plt.scatter(model_post_tilt_match[:, 0], model_post_tilt_match[:, 1], s=15, marker='x', label='Model')

plt.title("Post Tilt Optimisation")
plt.gca().set_aspect('equal')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#### Pysical model residuals  ################

plt.figure(figsize=(8, 8))
plt.quiver(model_match[:, 0], model_match[:, 1],
           res_x, res_y,
           color='red', angles='xy', scale_units='xy', scale=0.2) 
plt.scatter(x_raw, y_raw, s=5, color='black', alpha=0.2)
plt.title("Residuals From Data to Physical Model (5x Scale, Tilt Optimisation Included) ")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.gca().set_aspect('equal')
plt.show()

plt.figure(figsize=(8, 8))
plt.quiver(model_match[:, 0], model_match[:, 1],
           res_x, res_y,
           color='red', angles='xy', scale_units='xy', scale=0.1) 
plt.scatter(x_raw, y_raw, s=5, color='black', alpha=0.2)
plt.title("Residuals From Data to Physical Model (10x Scale) ")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.gca().set_aspect('equal')
plt.show()




### Residuals by altitude layers ##################

r_model = np.sqrt((model_match[:, 0] - x0_dome)**2 +
                  (model_match[:, 1] - y0_dome)**2)

unique_alts = np.sort(np.unique(altitudes))
layer_rms_pre = []

print("\nPRE-POLYNOMIAL RMS ERROR BY ARC LAYER (Altitude)")
print(f"{'Altitude (deg)':<15} | {'Points':<10} | {'RMS Error (px)':<15}")
print("-" * 45)

for alt in unique_alts:
    expected_r = f_best * np.radians(alt)  # expected radius for this altitude
    # Select points belonging to this altitude ring 
    mask = np.abs(r_model - expected_r) < (f_best * np.radians(1))
    if np.any(mask):
        diff_x = res_x[mask]
        diff_y = res_y[mask]

        rms_layer = np.sqrt(np.mean(diff_x**2 + diff_y**2))

        layer_rms_pre.append((alt, rms_layer))
        print(f"{alt:<15.1f} | {np.sum(mask):<10} | {rms_layer:<15.4f}")

alts_pre, errors_pre = zip(*layer_rms_pre)

plt.figure(figsize=(8, 5))

plt.errorbar(alts_pre, errors_pre,
             yerr=mean_stds,
             fmt='o-', capsize=5,
             label='Model RMS ± Centroid Uncertainty')
plt.title("Physical Model Residual Error vs Altitude")
plt.xlabel("Altitude Angle (Degrees)")
plt.ylabel("RMS Error (Pixels)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()




################## INVERSE CORRECTION ######################################################################


# New inverse residuals from data to model
res_x_inv = model_match[:, 0] - x_raw
res_y_inv = model_match[:, 1] - y_raw

poly = PolynomialFeatures(degree=3) # Polynomial features generator

# Fit polynomial on data positions
coords_poly = poly.fit_transform(np.column_stack((x_raw, y_raw)))

# Learn how to move data points into ideal model
model_inverse_res_x = LinearRegression().fit(coords_poly, res_x_inv)
model_inverse_res_y = LinearRegression().fit(coords_poly, res_y_inv)

def apply_inverse_correction(x_in, y_in):
    """Corrects raw data with a polynomial to shift to ideal projection"""
    pts = np.column_stack((x_in, y_in)) # Input = distorted image coords
    p_feat = poly.transform(pts) # Transform input coordinates into polynomial feature space for regression
    dx = model_inverse_res_x.predict(p_feat) # Predict correction offsets dx/dy from polynomial model
    dy = model_inverse_res_y.predict(p_feat)
    
    return x_in + dx, y_in + dy # Move data coords toward ideal model coords

x_test, y_test = apply_inverse_correction(x_raw, y_raw)

rms = np.sqrt(np.mean((x_test - model_match[:,0])**2 +
                      (y_test - model_match[:,1])**2))

print(f"Polynomial Corrected model RMS: {rms:.4f} px")
print("\nINVERSE POLYNOMIAL COEFFICIENTS (Data to Model)")
feature_names = poly.get_feature_names_out(['x', 'y'])

print(f"{'Term':<12} | {'X Shift Coeff':<15} | {'Y Shift Coeff':<15}")
print("-" * 50)

# Intercept (constant offset)
print(f"{'Intercept':<12} | {model_inverse_res_x.intercept_:15.4e} | {model_inverse_res_y.intercept_:15.4e}")

# All polynomial terms
for name, cx, cy in zip(feature_names[1:], 
                        model_inverse_res_x.coef_[1:], 
                        model_inverse_res_y.coef_[1:]):
    print(f"{name:<12} | {cx:15.4e} | {cy:15.4e}")



##################### INVERSE CORRECTION PLOTS ###############################################

##### Residuals ##########
x_corr_inv, y_corr_inv = apply_inverse_correction(x_raw, y_raw)

# Residuals AFTER inverse correction
res_x_inv = x_corr_inv - model_match[:, 0]
res_y_inv = y_corr_inv - model_match[:, 1]


# RESIDUALS BY ALTITUDE #####

r_model_new = np.sqrt((model_match[:, 0] - x0_dome)**2 +
                  (model_match[:, 1] - y0_dome)**2)

unique_alts = np.sort(np.unique(altitudes))
layer_rms = []

print("\nRMS ERROR BY ARC LAYER (Altitude)")
print(f"{'Altitude (deg)':<15} | {'Points':<10} | {'RMS Error (px)':<15}")
print("-" * 45)

for alt in unique_alts:
    expected_r = f_best * np.radians(alt)

    mask = np.abs(r_model_new - expected_r) < (f_best * np.radians(1))

    if np.any(mask):
        diff_x = res_x_inv[mask]
        diff_y = res_y_inv[mask]

        rms_layer = np.sqrt(np.mean(diff_x**2 + diff_y**2))

        layer_rms.append((alt, rms_layer))
        print(f"{alt:<15.1f} | {np.sum(mask):<10} | {rms_layer:<15.4f}")
        
unique_alts = np.sort(np.unique(altitudes))
layer_rms_pre = []
alts, rms_vals = zip(*layer_rms)
plt.figure(figsize=(8, 5))
plt.errorbar(alts, rms_vals,
             yerr=mean_stds,  
             fmt='o-',         
             capsize=5,
             label='RMS ± Centroid Uncertainty')

plt.xlabel("Altitude (degrees)")
plt.ylabel("RMS Error (pixels)")
plt.title("Polynomial Model Residual Error vs Altitude")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

##### Physical and Polynomial Model with Alt #####
plt.figure(figsize=(8, 5))
plt.errorbar(alts_pre, errors_pre,
             yerr=mean_stds, fmt='o-', capsize=5,
             label='Physical Model RMS ± Centroid Uncertainty')
plt.errorbar(alts, rms_vals,
             yerr=mean_stds, fmt='o-', capsize=5,
             label='Polynomial Model RMS ± Centroid Uncertainty')
plt.xlabel("Altitude (degrees)")
plt.ylabel("RMS Error (pixels)")
plt.title("Model Residual Error vs Altitude with Centroid Uncertainty")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

###### Residuals QUIVER PLot #######

plt.figure(figsize=(10, 8))
plt.quiver(x_raw, y_raw,
           res_x_inv, res_y_inv,
           angles='xy', scale_units='xy', scale=0.2, color='green')
plt.scatter(x_raw, y_raw, s=5, color='black', alpha=0.3)
plt.title("Residuals After Inverse Polynomial Correction (Data to Ideal Model, 5x Scale)")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.gca().set_aspect('equal')
plt.grid(True, alpha=0.3)

plt.show()

plt.figure(figsize=(10, 8))
plt.quiver(x_raw, y_raw,
           res_x_inv, res_y_inv,
           angles='xy', scale_units='xy', scale=0.1, color='green')
plt.scatter(x_raw, y_raw, s=5, color='black', alpha=0.3)
plt.title("Residuals After Inverse Polynomial Correction (Data to Ideal Model, 10x Scale)")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.gca().set_aspect('equal')
plt.grid(True, alpha=0.3)

plt.show()



##### MODEL PROGRESSION PLOT ########

x_corr_inv, y_corr_inv = apply_inverse_correction(x_raw, y_raw)
plt.figure(figsize=(10, 8))
# Raw data 
plt.scatter(x_raw, y_raw,
            s=60, facecolors='none', edgecolors='black', alpha=0.5,
            label='Original Data')

# Physical model (after f, rotation, tilt if included)
plt.scatter(model_match[:, 0], model_match[:, 1],
            s=40, marker='x', color='red',
            label='Physical Model')

# Inverse polynomial corrected data
plt.scatter(x_corr_inv, y_corr_inv,
            s=10, color='blue',
            label='Corrected (Inverse Polynomial)')
plt.title("Model Alignment: Data to Physical Model to Corrected Data")
plt.xlabel("Sensor X (Pixels)")
plt.ylabel("Sensor Y (Pixels)")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.tight_layout()
plt.gca().set_aspect('equal')
plt.grid(True, linestyle=':', alpha=0.3)
plt.show()

# Find error magnitudes
error_mag_phys = np.sqrt(res_x**2 + res_y**2)
error_mag_final = np.sqrt(res_x_inv**2 + res_y_inv**2)

plt.figure(figsize=(12, 5))

plt.hist(error_mag_phys, bins=30, alpha=0.5, label='Physical Model', color='red')
plt.hist(error_mag_final, bins=30, alpha=0.5, label='After Polynomial Correction', color='green')

plt.title("Residual Error Magnitude Distribution, Tilt Included")
plt.xlabel("Residual Error (pixels)")
plt.ylabel("Number of Points")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


####################################### IMAGE CORRECTION ###############################################################

#### Apply Image Correction ###########

input_path = r"C:\Users\ed\Downloads\ASC Calibration Image.png"
output_path = r"C:\Users\ed\Downloads\ASC_Corrected_Image.png"

def apply_full_image_correction(image_path):
    """Applying polynomial correction model to input image"""
    
    img = cv2.imread(image_path) # Loading the image with OpenCV
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting from BGR to RGB so that the colours arent shifted
    h, w = img_rgb.shape[:2] # Extracting image height and width.

    # Create a grid of every pixel coordinate in the destination (corrected) image.
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Flatten to 1D list for the polynomial model
    grid_x_flat = grid_x.ravel()
    grid_y_flat = grid_y.ravel()
    
    # Use the inverse correction model to find each pixels source in the uncorrected image
    map_x, map_y = apply_inverse_correction(grid_x_flat, grid_y_flat)

    # Reshape back to image dimensions and convert to float32 (required by OpenCV)
    map_x = map_x.reshape((h, w)).astype(np.float32)
    map_y = map_y.reshape((h, w)).astype(np.float32)

    # Interpolating pixel values, cv2.remap pulls pixels from the uncorrected image into their corrected positions
    corrected_img = cv2.remap(img_rgb, map_x, map_y, 
                              interpolation=cv2.INTER_LINEAR, # Interlinear used to create a full smooth image
                              borderMode=cv2.BORDER_CONSTANT, 
                              borderValue=(0, 0, 0))

    # Plot uncorrected vs corrected image
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Distorted Image (Raw image)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(corrected_img)
    plt.title(f"Corrected Image (Polynomial Degree {poly.degree})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Saving the result and convert back to BGR for OpenCV saving
    cv2.imwrite(output_path, cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR))
    print(f"Corrected image saved to: {output_path}")

# Run the correction
apply_full_image_correction(input_path)


######## POLYNOMIAL HEATMAP ############

resolution = 1000

# Grid across full image
x = np.linspace(0, WIDTH, resolution)
y = np.linspace(0, HEIGHT, resolution)
gx, gy = np.meshgrid(x, y)
# Turning grid 1D for correction
gx_flat = gx.ravel()
gy_flat = gy.ravel()

# Apply inverse correction
x_corr, y_corr = apply_inverse_correction(gx_flat, gy_flat)
dx = x_corr - gx_flat
dy = y_corr - gy_flat
mag = np.sqrt(dx**2 + dy**2)

# Mask for image outside the maximum data extent
r = np.sqrt((gx_flat - x0_dome)**2 + (gy_flat - y0_dome)**2)
r_max = np.max(r_raw)
mag[r > r_max] = np.nan
# Back to 2d matrix
mag = mag.reshape(gx.shape)

# Plot
plt.figure(figsize=(10, 8))
 # Limit heatmap to lower 95% of data to prevent outliers at the edges making graph less readable
vmax = np.nanpercentile(mag, 95)
im = plt.imshow(mag,
                extent=[0, WIDTH, 0, HEIGHT],
                origin='lower',
                aspect='equal',
                vmax=vmax)

plt.colorbar(im, label="Correction Magnitude (pixels)")

# Boundary circle
circle = plt.Circle((x0_dome, y0_dome), r_max,
                    color='red', fill=False,
                    linestyle='--', linewidth=2,
                    label='Data extent')
plt.gca().add_patch(circle)

# Calibration points
plt.scatter(x_raw, y_raw, s=5, color='white', alpha=0.3)

plt.title("Polynomial Distortion Map (Limited to Data Extent, including rotation)")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.legend()

plt.show()


##### OVERLAY PLOT ############

# Load image
img = cv2.imread(input_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

resolution = 1000
alpha_overlay = 0.6

# Create grid for image, I have to redo these steps to make sure the heatmap aligns with the image
x = np.linspace(0, w, resolution)
y = np.linspace(0, h, resolution)
gx, gy = np.meshgrid(x, y)
# Turning grid 1D for correction
gx_flat = gx.ravel()
gy_flat = gy.ravel()

# Apply inverse correction
x_corr, y_corr = apply_inverse_correction(gx_flat, gy_flat)
dx = x_corr - gx_flat
dy = y_corr - gy_flat
mag = np.sqrt(dx**2 + dy**2)

# Mask for image outside the maximum data extent
r = np.sqrt((gx_flat - x0_dome)**2 + (gy_flat - y0_dome)**2)
r_max = np.max(r_raw)
mag[r > r_max] = np.nan


# Back to 2d matrix
mag = mag.reshape(gx.shape)

plt.figure(figsize=(10, 8))
plt.imshow(img_rgb, extent=[0, w, h, 0], origin='lower')

# Heatmap overlay with lower 95% of data
vmax = np.nanpercentile(mag, 95)
im = plt.imshow(mag,
                extent=[0, w, 0, h],
                origin='lower',
                aspect='equal',
                alpha=alpha_overlay,
                vmax=vmax)

plt.colorbar(im, label="Correction Magnitude (pixels)")

# Data extent circle
circle = plt.Circle((x0_dome, y0_dome), r_max,
                    color='red', fill=False,
                    linestyle='--', linewidth=2)
plt.gca().add_patch(circle)

plt.title("Distortion Heatmap Overlay on Original Image")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.gca().set_aspect('equal')

plt.show()
