import numpy as np
import h5py
import os
from itertools import product
import glob
import re

def compute_Reff(image):
    ny, nx = image.shape
    y, x = np.indices((ny, nx))
    cy, cx = ny//2, nx//2

    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    flux = image.clip(min=0).ravel()
    r_flat = r.ravel()
    sort_idx = np.argsort(r_flat)
    flux_sorted = flux[sort_idx]
    r_sorted = r_flat[sort_idx]

    cumsum_flux = np.cumsum(flux_sorted)
    half_flux = cumsum_flux[-1]/2
    Reff_idx = np.searchsorted(cumsum_flux, half_flux)
    Reff = r_sorted[Reff_idx]
    return Reff

def create_library(z_object, id_object, snap_object, base_path, library_path, max_Reff_factor=10):
    os.makedirs(library_path, exist_ok=True)

    # Loop over all combinations of z_object and id_object
    for z_obj_i, id_obj_i, snap_obj_i in product(z_object, id_object, snap_object):
        file_path = f"{base_path}/Rvir_mocks_V2_{id_obj_i:05d}.h5"
        print("Processing:", file_path, "z =", z_obj_i)
        
        try:
            with h5py.File(file_path, "r") as f:
                data = f[f'xy/{snap_obj_i:d}/{z_obj_i:.1f}/image_full'][()]
                img = data[3, :, :]  # choose channel

            # Compute Reff
            Reff = compute_Reff(img)
            max_crop = int(max_Reff_factor * Reff)

            # Crop around centre
            ny, nx = img.shape
            cy, cx = ny//2, nx//2
            y1 = max(cy - max_crop, 0)
            y2 = min(cy + max_crop, ny)
            x1 = max(cx - max_crop, 0)
            x2 = min(cx + max_crop, nx)
            img_crop = img[y1:y2, x1:x2]

            # Save cropped image to library
            save_path = os.path.join(library_path, f"obj_{id_obj_i:05d}_z{z_obj_i:.2f}_{snap_obj_i:d}.h5")
            with h5py.File(save_path, "w") as f_out:
                f_out.create_dataset("image", data=img_crop, compression="lzf")
            print("Saved:", save_path)
        except:
            print('!')
        

base_path = '/dat/garreth/tmp'
library_path = './library'

files = glob.glob(f'{base_path}/Rvir_mocks_V2*.h5')
unique_ids = np.int32(sorted({re.search(r'_(\d+)\.h5$', f).group(1) for f in files}))

z_object = [0.05, 0.1, 0.2, 0.4]
id_object = list(unique_ids)
snap_object = [658, 791, 904]

create_library(z_object, id_object, snap_object, base_path, library_path)
