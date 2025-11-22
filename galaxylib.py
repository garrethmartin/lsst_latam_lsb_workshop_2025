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

    def create_canvas(self, shape, n_objects, seed=None, z_min=0.1, z_max=2.0):
        # generate canvas with random galaxies
        rng = np.random.default_rng(seed)
        canvas = np.zeros(shape, dtype=np.float32)

        files = [
            f for f in self.files
            if z_min <= self.extract_z(os.path.basename(f)) <= z_max
        ]


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
    
    def add_sky(self, image,
                *,
                sky_mag=22.5,               # sky surface brightness in mag/arcsec^2
                fluxmag0=6.309573448e10,    # reference flux for 0 mag in ADU
                pixscale=0.2,               # arcsec / pixel
                effective_gain=300,         # effective e-/ADU
                n_stars=100,
                star_mag_range=(18.0, 22.0), # star magnitudes (min, max)
                psf_sigma=2.5,
                sb_noise_mag=31.0,           # 3 sigma SB limit in mag/arcsec^2 for a 10"x10" aperture
                seed=None):
        """
        Produce a stacked-like image with:
          - sky level specified in mag/arcsec^2 (uses fluxmag0)
          - a small polynomial tilt in the background
          - stars drawn from a magnitude range
          - Poisson noise sampled in electrons using effective electron-per-ADU for the stack
          - Gaussian SB-limit noise corresponding to 3 sigma in a 10"x10" aperture

        Returns: (final_image_adus, background_adus, noise_map_adus)
        where noise_map_adus = final_image_adus - image_sky (image_sky is the PSF-smoothed scene)
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(seed)

        ny, nx = image.shape

        # mean sky in ADU/pixel
        mean_sky_adus = fluxmag0 * 10.0**(-0.4 * sky_mag) * (pixscale ** 2)
        
        # background with small polynomial tilt (absolute ADU coefficients)
        background = np.full((ny, nx), mean_sky_adus, dtype=float)
        y = np.linspace(-1.0, 1.0, ny)[:, None]
        x = np.linspace(-1.0, 1.0, nx)[None, :]
        poly_coeffs = {(1, 0): -0.03 * mean_sky_adus,  # ~ -3% tilt along x
                       (0, 1):  0.02 * mean_sky_adus}  # ~ +2% tilt along y
        for (i, j), c in poly_coeffs.items():
            background += c * (x ** i) * (y ** j)

        # -stars: convert magnitude range to ADU flux range for generate_stars API
        fmin = fluxmag0 * 10.0 ** (-0.4 * star_mag_range[1])
        fmax = fluxmag0 * 10.0 ** (-0.4 * star_mag_range[0])
        stars = self.generate_stars((ny, nx), n_stars=n_stars,
                                    flux_range=(fmin, fmax), seed=seed)

        scene = image + background + stars
        image_sky = gaussian_filter(scene, sigma=psf_sigma) # PSF convolution

        # gaussian noise from 3σ 10"x10" SB limit
        aperture_arcsec = 10.0
        sb_sigma_level = 3.0
        aperture_area = aperture_arcsec ** 2  # arcsec^2
        ap_flux = fluxmag0 * 10.0 ** (-0.4 * sb_noise_mag) * aperture_area  # ADU in aperture at sb_mag
        npix_ap = aperture_area / (pixscale ** 2)
        sigma_gauss_pixel = ap_flux / sb_sigma_level / np.sqrt(npix_ap)  # ADU per pixel (1-sigma)
        gauss_noise = rng.normal(loc=0.0, scale=sigma_gauss_pixel, size=(ny, nx))

        # poisson noise: sample in electrons using effective_gain = e-/ADU
        # expected electrons per pixel = image_sky (ADU) * (e-/ADU)
        lam_electrons = np.clip(image_sky * effective_gain, 0.0, None)
        poisson_e = rng.poisson(lam_electrons).astype(float)
        poisson_adu = poisson_e / effective_gain

        # compose final image
        final_image = poisson_adu + gauss_noise
        noise_map = final_image - image_sky

        return final_image, background, noise_map
