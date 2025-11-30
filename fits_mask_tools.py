import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, LogStretch, SqrtStretch, AsinhStretch
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from scipy.stats import norm
from ipywidgets import interact, FloatSlider, Dropdown, Button, HBox, VBox, IntSlider
from matplotlib.widgets import RectangleSelector
from skimage.measure import block_reduce
from IPython.display import clear_output, display
import pickle

class DisplayController:
    '''
    Class to manage display properties.
    '''
    def __init__(self, reference_image, stretch='linear', contrast=1.0, white_frac=1.0, scaling=99.0):
        self.reference_image = reference_image
        self.stretch = stretch
        self.contrast = contrast
        self.white_frac = white_frac
        self.scaling = scaling

        self._stretch_map = {
            'linear': LinearStretch(),
            'log': LogStretch(),
            'sqrt': SqrtStretch(),
            'asinh': AsinhStretch()
        }

    def update(self, stretch=None, contrast=None, white_frac=None, scaling=None):
        if stretch is not None:
            self.stretch = stretch
        if contrast is not None:
            self.contrast = contrast
        if white_frac is not None:
            self.white_frac = white_frac
        if scaling is not None:
            self.scaling = scaling

    def _compute_vmin_vmax_from_reference(self):
        # percentiles based on the stored image
        lower = (100.0 - self.scaling) / 2.0
        upper = 100.0 - lower
        vmin, vmax = np.nanpercentile(self.reference_image, [lower, upper])
        return vmin, vmax

    def get_norm(self, arr, use_reference=True):
        if use_reference:
            vmin, vmax = self._compute_vmin_vmax_from_reference()
        else:
            lower = (100.0 - self.scaling) / 2.0
            upper = 100.0 - lower
            vmin, vmax = np.nanpercentile(arr, [lower, upper])

        vmax_adj = vmin + self.white_frac * (vmax - vmin)
        vcenter = 0.5 * (vmin + vmax_adj)
        vhalf_range = 0.5 * (vmax_adj - vmin) / max(self.contrast, 1e-6)
        vmin_norm = vcenter - vhalf_range
        vmax_norm = vcenter + vhalf_range

        stretch = self._stretch_map.get(self.stretch, LinearStretch())
        try:
            return ImageNormalize(arr, vmin=vmin_norm, vmax=vmax_norm, stretch=stretch)
        except Exception:
            # fallback if something goes wrong
            return ImageNormalize(arr, vmin=np.nanmin(arr), vmax=np.nanmax(arr), stretch=LinearStretch())

class FitsViewer:
    '''
    image : 2D numpy array
        Image data to display.
    crop : tuple (y0, y1, x0, x1), optional
        Region to crop the image for display.
    figsize : tuple, optional
        Figure size for matplotlib display.
    '''
    def __init__(self, image, crop=None, figsize=(8, 8), display_controller=None):
        self.image_data = image
        if crop:
            self.image_data = self.image_data[crop[0]:crop[1], crop[2]:crop[3]]

        self.display_data = self.image_data.copy()

        # display controller
        if display_controller is None:
            self.display_controller = DisplayController(self.image_data)
        else:
            self.display_controller = display_controller

        # create figure and axes
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_axis_off()
        divider = make_axes_locatable(self.ax)
        self.cax = divider.append_axes("right", size="3%", pad=0.05)

        # initial display
        self.im = self.ax.imshow(self.display_data, origin='lower', cmap='gray', interpolation='nearest')
        norm = self.display_controller.get_norm(self.image_data, use_reference=True)
        try:
            self.im.set_norm(norm)
        except Exception:
            pass
        self.cbar = self.fig.colorbar(self.im, cax=self.cax)

        # launch interactive widgets
        self._create_widgets()

    def _create_widgets(self):
        self.stretch_widget = Dropdown(options=['linear', 'log', 'sqrt', 'asinh'], value=self.display_controller.stretch,
                                      description='stretch')
        self.contrast_slider = FloatSlider(value=self.display_controller.contrast, min=0.05, max=2.0, step=0.01,
                                          description='contrast')
        self.white_slider = FloatSlider(value=self.display_controller.white_frac, min=0.0, max=10.0, step=0.01,
                                        description='white')
        self.scaling_slider = FloatSlider(value=self.display_controller.scaling, min=90.0, max=100.0, step=0.05,
                                          description='scaling (%)')

        interact(self._update_from_widgets,
                 stretch_type=self.stretch_widget,
                 contrast=self.contrast_slider,
                 white_frac=self.white_slider,
                 scaling=self.scaling_slider)

    def _update_from_widgets(self, stretch_type='linear', contrast=1.0, white_frac=1.0, scaling=99.0):
        # update controller then refresh image
        self.display_controller.update(stretch=stretch_type, contrast=contrast, white_frac=white_frac, scaling=scaling)
        norm = self.display_controller.get_norm(self.image_data, use_reference=True)
        try:
            self.im.set_norm(norm)
            self.cbar.update_normal(self.im)
            self.fig.canvas.draw_idle()
        except Exception:
            pass
        
class BinnedFitsViewer(FitsViewer):
    '''
    Inherits FitsViewer.
    Creates slider that allows interactive rebinning of the displayed image.
    Updates FitsViewer display directly.
    '''
    def __init__(self, fv: FitsViewer, max_bin=20):
        """
        fv : existing FitsViewer instance
        max_bin : maximum binning factor for display
        """
        self.fv = fv
        self.full_image = fv.image_data.copy()
        self.display_controller = fv.display_controller
        self.im = fv.im
        self.cbar = fv.cbar
        self.ax = fv.ax
        self.fig = fv.fig

        # Start with no binning (factor=1)
        self.image_data = self.full_image.copy()
        self.display_data = np.ma.array(self.image_data)

        # Create a slider to control binning
        self.bin_slider = IntSlider(value=1, min=1, max=max_bin, step=1, description='Binning')
        self.bin_slider.observe(self._update_binning, names='value')
        display(self.bin_slider)

    def _update_binning(self, change):
        bin_factor = change['new']
        if bin_factor <= 1:
            binned = self.full_image
        else:
            binned = self._bin_image(self.full_image, bin_factor)

        # Update display image only
        self.image_data = binned
        self.display_data = np.ma.array(binned)

        norm = self.display_controller.get_norm(self.image_data, use_reference=True)
        self.im.set_data(self.display_data)
        try:
            self.im.set_norm(norm)
            self.cbar.update_normal(self.im)
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    @staticmethod
    def _bin_image(image, factor):
        """Bin the image by averaging over factor x factor blocks."""
        ny, nx = image.shape
        ny_binned = ny // factor
        nx_binned = nx // factor
        binned = image[:ny_binned*factor, :nx_binned*factor].reshape(ny_binned, factor, nx_binned, factor)
        return binned.mean(axis=(1,3))
    
    def save_state(self, filename='bfv_state.pkl'):
        state = {
            'full_image': self.full_image,
            'display_controller': {
                'stretch': self.display_controller.stretch,
                'contrast': self.display_controller.contrast,
                'white_frac': self.display_controller.white_frac,
                'scaling': self.display_controller.scaling
            },
            'current_bin': self.bin_slider.value
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f'State saved to {filename}')

    @staticmethod
    def load_state(filename='bfv_state.pkl', figsize=(8,8), max_bin=20):
        """
        Load saved state and return a fully interactive BinnedFitsViewer whose
        display (imshow + colourbar + widgets) reflects the saved binning and
        display parameters without then being reset to the original.
        """
        with open(filename, 'rb') as f:
            state = pickle.load(f)

        full_image = state['full_image']
        bin_factor = int(state.get('current_bin', 1))

        # compute the binned image (works for bin_factor == 1)
        if bin_factor <= 1:
            binned = full_image.copy()
        else:
            binned = BinnedFitsViewer._bin_image(full_image, bin_factor)

        # Create a fresh DisplayController bound to the binned image, then apply saved params.
        # This ensures get_norm() will use the binned image, not the original full image.
        dc = DisplayController(binned)
        dc.update(
            stretch=state['display_controller']['stretch'],
            contrast=state['display_controller']['contrast'],
            white_frac=state['display_controller']['white_frac'],
            scaling=state['display_controller']['scaling']
        )

        # Create a FitsViewer that is initialised from the binned image and the fresh controller.
        # This constructs figure, axes, imshow and colourbar already referencing 'binned'.
        fv_binned = FitsViewer(binned, figsize=figsize, display_controller=dc)

        # Now create the interactive BinnedFitsViewer from that FitsViewer.
        obj = BinnedFitsViewer(fv_binned, max_bin=max_bin)

        # Replace obj.full_image with the true full-resolution image so other features (zoom, mask) use it.
        obj.full_image = full_image.copy()

        # Make sure the display shows exactly the binned array and saved controller.
        obj.image_data = binned
        obj.display_data = np.ma.array(binned)
        obj.display_controller = dc  # ensure the object keeps the fresh controller

        # update the im, norm, colourbar to the saved settings
        norm = obj.display_controller.get_norm(obj.image_data, use_reference=True)
        obj.im.set_data(obj.display_data)
        try:
            obj.im.set_norm(norm)
            obj.cbar.update_normal(obj.im)
        except Exception:
            # Some mpl backends might raise; still proceed
            pass
        obj.fig.canvas.draw_idle()

        # set the slider to saved bin value without triggering another update
        try:
            obj.bin_slider.unobserve(obj._update_binning, names='value')
            obj.bin_slider.value = bin_factor
            obj.bin_slider.observe(obj._update_binning, names='value')
        except Exception:
            # if slider is missing or observe API differs, ignore silently
            pass

        return obj

class BackgroundInteractive:
    '''
    image : 2D numpy array
        image data.
    figsize : tuple, optional
        Figure size for display.
    zoom_size : int, optional
        Default size of zoomed-in panel.
    inset_frac : float, optional
        Size of inset relative to main plot.
    '''
    def __init__(self, image, figsize=(12, 6), zoom_size=50, inset_frac=0.35, display_controller=None):
        self.image = image
        self.zoom_size = zoom_size
        self.zoom_centre = None
        self.inset_frac = inset_frac
        self.bkg_map = None
        self.resid = None
        self.rect = None
        self.ax_inset = None

        # display controller: if none provided create one based on the full image
        if display_controller is None:
            self.display_controller = DisplayController(self.image)
        else:
            self.display_controller = display_controller

        self.fig, self.axs = plt.subplots(1, 2, figsize=figsize)
        self.fig.subplots_adjust(wspace=0)
        self.axs = self.axs.flatten()
        for ax in self.axs:
            ax.set_axis_off()

        # connect click on residual
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def _compute_background(self, box_size=128, filter_size=3, sigma=3.0, maxiters=5):
        sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)
        bkg_estimator = MedianBackground()
        bkg = Background2D(
            self.image,
            box_size=box_size,
            filter_size=filter_size,
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            edge_method='pad'
        )
        return bkg.background

    def _plot(self, box_size=128):
        self.last_box_size = box_size

        clear_output(wait=True)
        print(f'Running for box={box_size}...')

        self.bkg_map = self._compute_background(box_size)
        self.resid = self.image - self.bkg_map

        # determine residual scaling
        lower = (100.0 - 97.5) / 2.0
        upper = 100.0 - lower
        vmin_resid, vmax_resid = np.nanpercentile(self.resid, [lower, upper])

        # store so the inset can use identical scaling
        self.vmin_resid = vmin_resid
        self.vmax_resid = vmax_resid

        # background
        self.axs[0].cla()
        bkg_display = np.arcsinh(self.bkg_map)
        self.axs[0].imshow(bkg_display, origin='lower', cmap='viridis')
        self.axs[0].set_title(f'Background (box={box_size})')
        self.axs[0].set_axis_off()

        # residual
        self.axs[1].cla()
        self.axs[1].imshow(
            np.arcsinh(self.resid),
            origin='lower',
            cmap='Greys_r',
            vmin=self.vmin_resid,
            vmax=self.vmax_resid
        )
        self.axs[1].set_title('Residual (click to zoom)')
        self.axs[1].set_axis_off()

        # if a zoom was already selected, restore it
        if self.zoom_centre is not None:
            self._draw_zoom_elements()

        self.fig.canvas.draw_idle()
        print('Done.')

    def onclick(self, event):
        if event.inaxes != self.axs[1]:
            return
        self.zoom_centre = (int(event.xdata), int(event.ydata))
        self._draw_zoom_elements()

    def _draw_zoom_elements(self):
        if self.resid is None or self.zoom_centre is None:
            return

        x, y = self.zoom_centre
        s = self.zoom_size // 2
        ny, nx = self.resid.shape
        x0, x1 = max(0, x - s), min(nx, x + s)
        y0, y1 = max(0, y - s), min(ny, y + s)

        sub = self.resid[y0:y1, x0:x1]

        # remove any old rectangle/inset
        if self.rect:
            self.rect.remove()
        if self.ax_inset:
            self.ax_inset.remove()

        # draw rectangle on residual
        self.rect = Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='red', facecolor='none', lw=2)
        self.axs[1].add_patch(self.rect)

        # inset location (top-left)
        self.ax_inset = inset_axes(
            self.axs[1],
            width=f"{int(self.inset_frac*100)}%",
            height=f"{int(self.inset_frac*100)}%",
            loc='upper left',
            borderpad=1
        )

        # use same mapping as parent residual (arcsinh + same vmin/vmax)
        self.ax_inset.imshow(
            np.arcsinh(sub),
            origin='lower',
            cmap='Greys',
            vmin=self.vmin_resid,
            vmax=self.vmax_resid
        )
        self.ax_inset.set_xticks([])
        self.ax_inset.set_yticks([])

        self.fig.canvas.draw_idle()

    def _update_zoom(self, zoom_size):
        self.zoom_size = zoom_size
        if self.zoom_centre is not None:
            self._draw_zoom_elements()

    def interact(self):
        box_slider = IntSlider(value=128, min=32, max=1024, step=32, description='box size', continuous_update=False)
        zoom_slider = IntSlider(value=self.zoom_size, min=100, max=1000, step=100, description='zoom size', continuous_update=True)

        interact(self._plot, box_size=box_slider)
        zoom_slider.observe(lambda change: self._update_zoom(change['new']), names='value')

        display(VBox([zoom_slider]))

    def plot_residual_hist(self, tile_size=50, bins=50, clip_sigma=3):
        if self.resid is None:
            print("Residual not computed yet. Run _plot() first.")
            return

        box_size = getattr(self, 'last_box_size', None)

        residual = self.resid
        img_sky = self.image
        ny, nx = residual.shape
        tile_stds = []

        # Compute standard deviations in tiles
        for y0 in range(0, ny, tile_size):
            for x0 in range(0, nx, tile_size):
                y1 = min(y0 + tile_size, ny)
                x1 = min(x0 + tile_size, nx)
                tile = img_sky[y0:y1, x0:x1]
                tile_stds.append(np.std(tile))

        sigma_bg = np.median(tile_stds)

        data = residual.ravel()

        if clip_sigma is not None:
            mean = np.mean(data)
            std = np.std(data)
            mask = (data > mean - clip_sigma * std) & (data < mean + clip_sigma * std)
            clipped_data = data[mask]
        else:
            clipped_data = data
            mean = np.mean(data)
            std = np.std(data)

        plt.figure(figsize=(5, 4))
        counts, bins_edges, _ = plt.hist(clipped_data, bins=bins, density=True, edgecolor='black', alpha=0.7, label='Residuals')

        x = np.linspace(bins_edges[0], bins_edges[-1], 200)
        plt.plot(x, norm.pdf(x, 0, sigma_bg), 'r--', lw=2, label='Expected Gaussian noise')

        plt.xlabel("Residual value")
        plt.ylabel("Normalized count")
        plt.title(rf'Residuals (box={box_size})')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.show()

        
class ImageCropper:
    """
    Interactive square crop tool for extracting a region from an image.
    Returns both original and optionally binned data.
    """
    def __init__(self, image, bin_size=2):
        self.image = image
        self.bin_size = bin_size
        self.crop = None

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(np.arcsinh(image), origin='lower', cmap='gray')
        self.ax.set_title('Drag to select a region')
        
        self.RS = RectangleSelector(self.ax, self.on_select, 
                                    interactive=True,
                                    useblit=True,
                                    button=[1],  # left click only
                                    minspanx=1, minspany=1,
                                    spancoords='pixels')
        plt.show()

    def on_select(self, eclick, erelease):
        x0, y0 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x1, y1 = int(round(erelease.xdata)), int(round(erelease.ydata))
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        
        self.crop = self.image[y0:y1, x0:x1]
        print(f'Selected region: x={x0}:{x1}, y={y0}:{y1}, shape={self.crop.shape}')

    def get_crops(self):
        if self.crop is None:
            raise ValueError("No region selected yet.")
        binned_crop = block_reduce(self.crop, block_size=(self.bin_size, self.bin_size), func=np.mean)
        return self.crop, binned_crop

class MaskPainter:
    '''
    Interactive mask painting tool for a FitsViewer instance.

    fv : FitsViewer instance
        The image viewer object from which to inherit stretch, scaling, etc.
    brush_size : int, optional
        Default radius of brush in pixels.
    figsize : tuple, optional
        Figure size.
    '''

    def __init__(self, fv: FitsViewer, brush_size=10, figsize=(8, 8), display_controller=None):
        self.fv = fv
        self.full_image = fv.image_data.copy()
        self.full_ny, self.full_nx = self.full_image.shape

        self.zoom_box_size = 50
        self.zoom_centre = None
        self.zoomed = False

        self.image_data = self.full_image.copy()
        self.current_mask = np.zeros_like(self.image_data, dtype=np.uint8)
        self.full_masks = []
        self.mask_colors = []

        # Combined overlay for all committed masks
        self.full_overlay = np.zeros_like(self.full_image, dtype=np.uint8)

        self.brush_size = brush_size
        self.painting = False

        # Use provided display_controller or take from FitsViewer
        if display_controller is None:
            self.display_controller = getattr(fv, 'display_controller', DisplayController(self.full_image))
        else:
            self.display_controller = display_controller

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_axis_off()
        self.im = self.ax.imshow(self.image_data, origin='lower', cmap='gray', interpolation='nearest')
        try:
            self.im.set_norm(self.display_controller.get_norm(self.image_data))
        except Exception:
            pass

        self.current_overlay_im = None
        self.overlay_im = None  # single overlay for all committed masks

        self.next_button = Button(description='next mask', button_style='info')
        self.next_button.on_click(self._next_mask)
        self.save_button = Button(description='save masks', button_style='success')
        self.save_button.on_click(self._save_masks)
        self.reset_button = Button(description='return to original', button_style='warning')
        self.reset_button.on_click(self._reset_to_full)

        self.zoom_slider = IntSlider(value=self.zoom_box_size, min=50, max=1000, step=10, description='zoom size', continuous_update=False)
        self.zoom_slider.observe(self._on_zoom_slider_change, names='value')
        self.brush_slider = IntSlider(value=self.brush_size, min=1, max=100, step=1, description='Brush size', continuous_update=True)
        self.brush_slider.observe(self._on_brush_size_change, names='value')
        
        self.feature_type_dropdown = Dropdown(
            options=['Stellar Stream', 'Tidal Tail', 'Plume', 'Shell', 'Tidal Bridge', 'Merger Remnant'],
            value='Stellar Stream', description='Feature type:')

        display(HBox([self.next_button, self.save_button, self.reset_button, self.feature_type_dropdown, self.zoom_slider,
                      self.brush_slider]))

        os.makedirs('./masks', exist_ok=True)

        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

        self.fig.canvas.draw_idle()
        print('MaskPainter ready — click to create a zoom crop, paint, then commit.')
        
    def _on_brush_size_change(self, change):
        self.brush_size = change['new']

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        xpix = int(round(event.xdata))
        ypix = int(round(event.ydata))

        if not self.zoomed:
            self._start_zoom_at(xpix, ypix)
            return

        full_x = int(round(self.view_x0 + xpix))
        full_y = int(round(self.view_y0 + ypix))

        self.painting = True
        self.zoom_slider.disabled = True
        self._paint_at(full_x, full_y)

    def _on_motion(self, event):
        if not self.painting or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        xpix = int(round(event.xdata))
        ypix = int(round(event.ydata))

        full_x = int(round(self.view_x0 + xpix)) if self.zoomed else xpix
        full_y = int(round(self.view_y0 + ypix)) if self.zoomed else ypix

        self._paint_at(full_x, full_y)

    def _on_release(self, event):
        if self.painting:
            self.painting = False

    def _paint_at(self, xpix, ypix):
        if not self.zoomed:
            return
        x_local = xpix - self.view_x0
        y_local = ypix - self.view_y0

        h, w = self.current_mask.shape
        if x_local < 0 or y_local < 0 or x_local >= w or y_local >= h:
            return

        yy, xx = np.ogrid[:h, :w]
        mask_circle = (yy - y_local) ** 2 + (xx - x_local) ** 2 <= self.brush_size ** 2
        self.current_mask[mask_circle] = 1

        if self.current_overlay_im is None:
            self.current_overlay_im = self.ax.imshow(
                np.ma.masked_where(self.current_mask == 0, self.current_mask),
                origin='lower', cmap='Reds', alpha=0.5, interpolation='nearest', extent=(0, w, 0, h)
            )
        else:
            self.current_overlay_im.set_data(np.ma.masked_where(self.current_mask == 0, self.current_mask))

        self.fig.canvas.draw_idle()

    def _start_zoom_at(self, xpix, ypix):
        self.zoom_centre = (xpix, ypix)
        self._apply_zoom()

    def _apply_zoom(self, size=None):
        if self.zoom_centre is None:
            return
        if self.painting:
            print('Cannot change zoom while painting. Commit the mask first.')
            return

        if size is None:
            size = int(self.zoom_slider.value)
        half = size // 2
        x, y = self.zoom_centre

        x0 = max(0, x - half)
        x1 = min(self.full_nx, x + half)
        y0 = max(0, y - half)
        y1 = min(self.full_ny, y + half)

        self.view_x0, self.view_x1 = x0, x1
        self.view_y0, self.view_y1 = y0, y1
        self.zoom_box_size = size
        self.zoomed = True

        self.image_data = self.full_image[y0:y1, x0:x1]
        self.current_mask = np.zeros_like(self.image_data, dtype=np.uint8)

        self.im.set_data(self.image_data)

        h, w = self.image_data.shape
        self.im.set_extent((0, w, 0, h))
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)

        if self.current_overlay_im:
            self.current_overlay_im.remove()
            self.current_overlay_im = None

        self._update_zoom_overlay()
        self.fig.canvas.draw_idle()
        print(f'Zoomed to size={size} centred on {self.zoom_centre}.')

    def _on_zoom_slider_change(self, change):
        if self.painting:
            self.zoom_slider.value = self.zoom_box_size
            print('Cannot change zoom size while painting. Commit the mask first.')
            return

        new_size = int(change['new'])
        if self.zoomed:
            self.current_mask = np.zeros_like(self.current_mask)
            if self.current_overlay_im is not None:
                self.current_overlay_im.remove()
                self.current_overlay_im = None
            self._apply_zoom(size=new_size)
        else:
            self.zoom_box_size = new_size

    def _next_mask(self, b):
        if not self.zoomed:
            print('No active crop to commit from. Click to create a crop first.')
            return
        if not np.any(self.current_mask):
            print('Local mask empty; nothing to commit.')
            return

        y0, y1 = self.view_y0, self.view_y1
        x0, x1 = self.view_x0, self.view_x1
        
        mask_label = self.feature_type_dropdown.value.strip()

        full_mask = np.zeros_like(self.full_image, dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = self.current_mask
        
        self.full_masks.append({
            'mask': full_mask,
            'label': mask_label
        })

        mask_id = len(self.mask_colors) + 1
        self.full_overlay[full_mask > 0] = mask_id
        color = np.random.rand(3,)
        self.mask_colors.append(color)

        self.current_mask = np.zeros_like(self.image_data, dtype=np.uint8)
        if self.current_overlay_im:
            self.current_overlay_im.remove()
            self.current_overlay_im = None

        self.zoom_slider.disabled = False
        self._update_zoom_overlay()
        print(f'Committed mask #{len(self.full_masks)}.')

    def _update_zoom_overlay(self):
        if self.zoomed:
            y0, y1 = self.view_y0, self.view_y1
            x0, x1 = self.view_x0, self.view_x1
            cropped_overlay = self.full_overlay[y0:y1, x0:x1]
            cmap = mcolors.ListedColormap([[0, 0, 0]] + list(self.mask_colors))
            if self.current_overlay_im is None:
                self.current_overlay_im = self.ax.imshow(
                    np.ma.masked_where(cropped_overlay == 0, cropped_overlay), origin='lower', cmap=cmap, alpha=0.35,
                    interpolation='nearest', extent=(0, cropped_overlay.shape[1], 0, cropped_overlay.shape[0])
                )
            else:
                self.current_overlay_im.set_data(np.ma.masked_where(cropped_overlay == 0, cropped_overlay))
        else:
            if np.any(self.full_overlay):
                cmap = mcolors.ListedColormap([[0, 0, 0]] + list(self.mask_colors))
                if self.overlay_im is None:
                    self.overlay_im = self.ax.imshow(
                        np.ma.masked_where(self.full_overlay == 0, self.full_overlay), origin='lower', cmap=cmap, alpha=0.35,
                        interpolation='nearest', extent=(0, self.full_nx, 0, self.full_ny)
                    )
                else:
                    self.overlay_im.set_data(np.ma.masked_where(self.full_overlay == 0, self.full_overlay))

        self.fig.canvas.draw_idle()

    def _reset_to_full(self, b=None):
        if self.painting:
            print('Cannot return to full image while painting. Commit or stop painting first.')
            return

        if self.current_overlay_im:
            self.current_overlay_im.remove()
            self.current_overlay_im = None
        self.current_mask = np.zeros_like(self.current_mask)

        self.image_data = self.full_image.copy()
        self.zoom_centre = None
        self.zoomed = False
        self.view_x0, self.view_x1 = 0, self.full_nx
        self.view_y0, self.view_y1 = 0, self.full_ny

        self.im.set_data(self.image_data)
        self.im.set_extent((0, self.full_nx, 0, self.full_ny))
        self.ax.set_xlim(0, self.full_nx)
        self.ax.set_ylim(0, self.full_ny)

        self.zoom_slider.disabled = False
        self._update_zoom_overlay()
        print('Returned to full image (local mask discarded).')

    def _save_masks(self, b):
        import h5py

        if len(self.full_masks) == 0:
            print('No masks to save.')
            return

        masks_data = []

        for i, entry in enumerate(self.full_masks):
            full_mask = entry['mask']
            label = entry['label']
            
            y_indices, x_indices = np.where(full_mask > 0)
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue

            y0, y1 = y_indices.min(), y_indices.max() + 1
            x0, x1 = x_indices.min(), x_indices.max() + 1

            centre_y = (y0 + y1) // 2
            centre_x = (x0 + x1) // 2
            
            mask_dict = {
                'image_data': self.full_image[y0:y1, x0:x1],
                'mask': full_mask[y0:y1, x0:x1],
                'centre': (centre_x, centre_y),
                'corners': (x0, x1, y0, y1),
                'label': label
            }
            masks_data.append(mask_dict)

        # Save as HDF5
        with h5py.File('./masks/masks_data.hdf5', 'w') as f:
            for i, md in enumerate(masks_data):
                grp = f.create_group(f'mask_{i:04d}')
                grp.create_dataset('image_data', data=md['image_data'])
                grp.create_dataset('mask', data=md['mask'])
                grp.attrs['centre'] = md['centre']
                grp.attrs['corners'] = md['corners']
                grp.attrs['label'] = md['label']

        print(f'Saved {len(masks_data)} masks with associated cutouts to ./masks/masks_data.hdf5')

    @staticmethod
    def plot_saved_masks(hdf5_path, ncols=3, use_log=True):
        import h5py

        dict_masks = {}
        with h5py.File(hdf5_path, 'r') as f:
            keys = list(f.keys())
            n = len(keys)
            nrows = int(np.ceil(n / ncols))

            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
            axes = axes.flatten()

            for ax, name in zip(axes, keys):
                
                grp = f[name]
                
                img = f[f'{name}/image_data'][()]
                mask = f[f'{name}/mask'][()].astype(bool)
                
                label = grp.attrs.get('label', 'unlabelled')
                
                img_plot = np.log10(img) if use_log else img
                img_masked = np.ma.masked_where(~mask, img_plot)

                ax.imshow(img_plot, origin='lower', cmap='gray')
                ax.imshow(img_masked, origin='lower', cmap='viridis')
                ax.contour(mask.astype(float), levels=[0.5], colors='red', linewidths=2)

                flux = np.sum(img[mask])
                area = np.count_nonzero(mask)

                ax.set_title(f"{name}, {label}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

                dict_masks[name] = {
                    'masked_data': np.ma.masked_where(~mask, img),
                    'area': area,
                    'total_flux': flux,
                    'feature_label': label
                }

            for ax in axes[n:]:
                ax.axis('off')

            plt.tight_layout()
            plt.show()

        return dict_masks
