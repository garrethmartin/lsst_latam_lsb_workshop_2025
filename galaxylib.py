import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy import fftpack
from scipy.ndimage import gaussian_filter

class GalaxyLibrary:
    def __init__(self, library_path):
        # store library path
        self.library_path = library_path
        self.files = self._scan_library()

    def _scan_library(self):
        # find all h5 files
        return [os.path.join(self.library_path, f)
                for f in os.listdir(self.library_path)
                if f.endswith('.h5')]

    @staticmethod
    def extract_z(fname):
        # extract redshift from filename
        m = re.search(r"z([0-9.]+)", fname)
        return float(m.group(1)) if m else None

    def load_and_plot(self, plot=None):
        # optionally plot files matching given redshift
        if plot is None:
            return self.files

        zt = float(plot)
        for fp in self.files:
            zf = self.extract_z(os.path.basename(fp))
            if zf is None or not np.isclose(zf, zt, atol=1e-4):
                continue

            print(f"plotting: {os.path.basename(fp)} (z={zf:.2f})")
            with h5py.File(fp, 'r') as f:
                for name, dset in f.items():
                    arr = np.array(dset)
                    plt.figure()
                    plt.imshow(np.log10(arr), origin='lower', cmap='Greys')
                    plt.title(f"{os.path.basename(fp)}:{name}")

        plt.show()
        return self.files

    @staticmethod
    def inject_image(canvas, img, rng=None, x0=None, y0=None):
        # add image into canvas
        if rng is None:
            rng = np.random.default_rng()

        ny, nx = canvas.shape
        iy, ix = img.shape

        if x0 is None:
            x0 = rng.integers(ix//2, nx - ix//2)
        if y0 is None:
            y0 = rng.integers(iy//2, ny - iy//2)

        x1c = max(x0 - ix//2, 0)
        y1c = max(y0 - iy//2, 0)
        x2c = min(x1c + ix, nx)
        y2c = min(y1c + iy, ny)

        x1i = max(0, -(x0 - ix//2))
        y1i = max(0, -(y0 - iy//2))
        x2i = x1i + (x2c - x1c)
        y2i = y1i + (y2c - y1c)

        canvas[y1c:y2c, x1c:x2c] += img[y1i:y2i, x1i:x2i]

    def create_canvas(self, shape, n_objects, seed=None, z_min=0.1):
        # generate canvas with random galaxies
        rng = np.random.default_rng(seed)
        canvas = np.zeros(shape, dtype=np.float32)

        files = [f for f in self.files
                 if self.extract_z(os.path.basename(f)) >= z_min]

        if len(files) < n_objects:
            raise ValueError(f"not enough galaxies with z >= {z_min}")

        chosen = rng.choice(files, size=n_objects, replace=False)

        for fp in chosen:
            with h5py.File(fp, 'r') as f:
                img = f['image'][()]
            self.inject_image(canvas, img, rng)

        return canvas

    def inject_specific(self, canvas, object_positions, scale=1.0, flux_scale=1.0):
        # inject named objects at given coordinates
        for name, x, y in object_positions:
            fp = os.path.join(self.library_path, name)
            if not os.path.exists(fp):
                print(f"warning: {name} not found")
                continue

            with h5py.File(fp, 'r') as f:
                img = f['image'][()]

            if scale != 1.0:
                img = zoom(img, scale, order=1)
            if flux_scale != 1.0:
                img = img * flux_scale

            self.inject_image(canvas, img, x0=x, y0=y)

        return canvas

    def make_gaussian_field(self, shape, power=-3.0, seed=None):
        # gaussian random field
        rng = np.random.default_rng(seed)
        ny, nx = shape
        ky = np.fft.fftfreq(ny)[:, None]
        kx = np.fft.fftfreq(nx)[None, :]
        k = np.sqrt(kx**2 + ky**2)
        k[0, 0] = 1.0
        amp = k**(power/2.0)
        phases = rng.normal(size=(ny, nx)) + 1j*rng.normal(size=(ny, nx))
        field = np.fft.ifft2(amp * phases).real
        field -= field.mean()
        field /= field.std()
        return field

    def generate_stars(self, shape, n_stars=100, flux_range=(50, 200), alpha=2.0, seed=None):
        # sample star fluxes on a power law
        rng = np.random.default_rng(seed)
        ny, nx = shape
        fmin, fmax = flux_range
        u = rng.random(n_stars)
        fluxes = ((fmax**(1-alpha) - fmin**(1-alpha)) * u + fmin**(1-alpha))**(1/(1-alpha))
        stars = np.zeros(shape)
        for flux in fluxes:
            y0 = rng.integers(0, ny)
            x0 = rng.integers(0, nx)
            stars[y0, x0] += flux
        return stars

    def add_sky(self, image, *, mean_sky_adus=10.0,
                 poly_coeffs=(0.,0.,0.,0.,0.,0.),
                 additional_noise_sigma=0.5,
                 n_stars=100, star_flux_range=(50,200),
                 psf_sigma=2.5, gain=50., seed=None):
        # add background and stars
        ny, nx = image.shape
        y = np.linspace(-1, 1, ny)[:, None]
        x = np.linspace(-1, 1, nx)[None, :]
        
        background = np.full((nx, ny), mean_sky_adus, dtype=float)
        
        for (i, j), c in poly_coeffs.items():
            background += c * x**i * y**j
            
        stars = self.generate_stars((ny, nx), n_stars=n_stars,
                                    flux_range=star_flux_range, seed=seed)
        
        image_sky = gaussian_filter(image+background+stars, sigma=psf_sigma)

        noise = np.random.poisson(image_sky/gain).astype(float)
        noise += np.random.normal(scale=additional_noise_sigma, size=image.shape)
        
        return image_sky + noise, background, noise