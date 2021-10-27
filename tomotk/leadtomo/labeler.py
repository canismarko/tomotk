from functools import partial
import datetime as dt
import multiprocessing as mp
import logging
import math
import enum
from pathlib import Path
from typing import Union

from tqdm import tqdm
import h5py
import skimage
import numpy as np
from tomopy.misc.morph import downsample
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass
import polarTransform
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import unsharp_mask, threshold_yen, threshold_otsu, threshold_li, threshold_triangle, rank
from skimage.morphology import white_tophat, black_tophat, ball, disk, square, dilation, erosion, opening, closing

from tomotk import exceptions


Uri = Union[str, Path]


log = logging.getLogger(__name__)


class Labels(enum.IntEnum):
    BACKGROUND = 0
    PORE = 1
    ACTIVE_MATERIAL = 2
    FREE_LEAD = 3
    GRID = 4


def label_cylinder(hdf_uri: Uri, src_ds: str="volume", dest_ds: str="volume_labels", overwrite: bool=False):
    """Calculate labels for a volume stored in an HDF5 file.
    
    hdf_uri
      The HDF5 file to open.
    src_ds
      The path to the dataset with volume density data.
    dest_ds
      The path to the dataset that will hold the volume data.
    overwrite
      If true, *dest_ds* will be overwritten if it exists.

    """
    start = dt.datetime.now()
    # Create a labeler to do that labeling
    labeler = CylinderLabeler(max_shape=500)
    with h5py.File(hdf_uri, mode='r+') as h5fp:
        # Check now to make sure we're not going to fail to write at the end
        if dest_ds in h5fp.keys() and not overwrite:
            # Dataset exists but is not to be overwritten
            msg = (f"Found existing dataset '{hdf_uri}:{dest_ds}'. "
                   "Consider using ``overwrite=True`` if you're sure "
                   "this is the correct destination.")
            log.error(msg)
            raise exceptions.HDFDatasetExists(msg)
        # Do the actual labeling
        vol = h5fp[src_ds]
        labels = labeler.label(vol)
        # Overwrite the data if necessary
        if dest_ds in h5fp.keys() and overwrite:
            log.info("Overwriting dataset '%s' in file '%s'.", dest_ds, hdf_uri)
            del h5fp[dest_ds]
        # Save the result
        log.debug("Saving labels to HDF5 file %s:%s", hdf_uri, dest_ds)
        h5fp.create_dataset(name=dest_ds, dtype="uint8", data=labels)
    # Log to time it took to label this tomogram
    delta = dt.datetime.now() - start
    log.info(f"Labeled '{hdf_uri}:{src_ds}' in {str(delta)}.")


def polar_hull(slc):
    """Find the largest convex region using a polar transform."""
    old_dtype = slc.dtype
    # Convert to polar coordinates
    slc = slc.astype('uint8') * 256
    center = center_of_mass(slc)
    center_is_valid = not np.any(np.isnan(center))
    if not center_is_valid:
        log.warning("Unvalid center by center of mass method. "
                    "Defaulting to center of image")
        center = None
    log.debug("Performing polar transform")
    pim, ptSettings = polarTransform.convertToPolarImage(slc, center=center)
    log.debug("Applying anisotropic morphology filters to polar image.")
    pim = pim.astype('bool')
    pim = ndimage.median_filter(pim, size=9)
    pim = closing(pim, selem=np.ones((10, 1)))
    pim = opening(pim, selem=np.ones((1, 10)))
    # Mesh grid in order to compare edge positions
    my, mx = np.mgrid[:pim.shape[0],:pim.shape[1]]
    # Find the outside of the shape in the x direction
    log.debug("Calculating hull boundary")
    edge_r_idx = pim.shape[1] - np.argmax(pim[:,::-1], axis=1)
    edge_r_idx[edge_r_idx==pim.shape[1]] = 0
    edge_r = mx < edge_r_idx[:,np.newaxis]
    edge_r = ndimage.median_filter(edge_r, size=9)
    log.debug("Performing inverse polar transform")
    hull, uptSetting = polarTransform.convertToCartesianImage(edge_r.astype('uint8')*256, settings=ptSettings)
    return hull.astype(old_dtype)


def _label_pores_slice(slc_fg, dtype="uint8"):
    # Agressive denoise and sharpening
    sigma_est = np.mean(estimate_sigma(slc_fg))
    electrode = polar_hull(slc_fg)
    pores = np.logical_and(~slc_fg, electrode)
    return pores.astype(dtype)


class CylinderLabeler():
    """Takes a reconstructed tomogram and assign labels to the voxels.
    
    Specifically, this labeler is optimized to work on cylindrical
    operando lead-acid cells, where the grid, active material and
    separator are all stacked vertically (axis 0).
    
    """
    lead_profile = None
    bg_threshold = math.nan
    lead_threshold = math.nan
    downsample_levels = ()
    
    def __init__(self, max_shape=1024, max_counts=1e4):
        """Dimensions larger the *max_shape* will be downsampled.
        
        Values for the line profile larger than *max_counts* will be
        reduced then enlarged.
        
        Both *max_shape* and *max_counts* can be used to reduce the
        memory requirements, at the expense of precision. These values
        to do not affect the shape of the final result, only
        intermediate arrays.
        
        Parameters
        ==========
        
        """
        self.max_shape = max_shape
        self.max_counts = max_counts
    
    def downsample(self, vol):
        """Automatically convert the volume to manageable size."""
        levels = []
        for axis, shape in enumerate(vol.shape):
            level = int(np.log2(math.ceil(shape/self.max_shape)))
            levels.append(level)
            log.debug("Downsampling axis %d (level = %d)", axis, level)
            if level > 0:
                vol = downsample(vol, level=level, axis=axis)
        return vol
    
    def label(self, vol, passes=2):
        """Assign labels to the given volume."""
        # Convert HDF datasets to numpy arrays
        vol = vol[()]
        # Downsample to make it faster to do thresholding
        small_vol = self.downsample(vol)
        # Do the labeling
        for pass_ in range(passes):
            log.debug("Starting labeler pass #%d...", pass_)
            # Apply slice mask from previous thresholding pass
            mask = self.active_material_mask()
            subvol = small_vol[mask]
            # Do thresholding on the masked data
            self.update_thresholds(subvol)
            # Calculate and save a new slice profile for the next pass
            self.update_profile(small_vol)
        # Calculate the final labels from the full volume
        self.update_profile(vol)
        labels = self.label_volume(vol)
        # Beam hardening makes some of the grid look like active material, so fix it
        grid_mask = np.expand_dims(self.grid_mask(), axis=(1, 2))
        is_grid = np.logical_and(grid_mask, labels > Labels.BACKGROUND)
        labels[is_grid] = Labels.GRID
        # 3D labeling does not work well for finding free lead, so let's do that by 2D thresholding
        free_lead = self.label_lead_2d(vol, labels)
        labels[free_lead] = Labels.FREE_LEAD
        # Now isolate the pores from the regular background
        pores = self.label_pores(labels > Labels.BACKGROUND)
        log.debug("Highlighting pores in labels volume.")
        pores = pores.astype('bool', copy=False)
        labels[pores] = Labels.PORE
        log.debug("End of CylinderLabeler.label()")
        return labels

    def label_pores(self, vol_fg):
        """Label the internal pores within the electrode.
        
        Parameters
        ==========
        vol_fg : ndarray
          A 3D array where non-zero values are the foreground
          (i.e. active material/free-lead/etc)
        
        Returns
        =======
        pores : ndarray
          A 3D array with shape of *vol_fg*, where nonzero values
          correspond to pores within the electrode.
        
        """
        desc = "Extracting pores"
        out = np.zeros_like(vol_fg, dtype="uint8")
        label_f = partial(_label_pores_slice, dtype=out.dtype)
        ncore = max(mp.cpu_count()-2, 1)
        log.debug(f"Starting pore labeling with {ncore} cores...")
        with mp.Pool(ncore) as p:
            map_it = p.imap(label_f, vol_fg, chunksize=32)
            for slc_idx, label_slc in enumerate(tqdm(map_it, total=vol_fg.shape[0], desc=desc, unit="slc")):
                out[slc_idx] = label_slc
        log.debug(f"Finished labeling pores")
        return out
    
    def label_lead_2d(self, vol, labels):
        """Use 2D thresholding to try and identify free lead.
        
        Parameters
        ==========
        vol
          3D array with density data in (z, y, x) order.
        labels
          3D array matching shape of *vol*, with initial guess at
          labels.
        
        Returns
        =======
        lead_labels
          3D array matching *labels* but of boolean dtype where true
          values are identified as free lead.
        
        """
        log.debug("Labeling free lead...")
        # Beam hardening makes labeling at the edge better by 3D
        # thresholding and in the middle better by 2D thresholding, so
        # let's figure that out
        yy, xx = np.mgrid[slice(vol.shape[1]), slice(vol.shape[2])]
        yy = 2 * 1.25 * (yy / np.max(yy) - 0.5)
        xx = 2 * 1.25 * (xx / np.max(xx) - 0.5)
        radii = np.sqrt(xx**2 + yy**2)
        radii[radii > 1] = 1
        # Do labeling for each slice
        lead_labels = []
        for idx, (slc, lbl) in enumerate(tqdm(zip(vol, labels), total=vol.shape[0], desc="Labeling free lead", unit="slc")):
            th_low, th_high = skimage.filters.threshold_multiotsu(slc, classes=3)
            slc_lead = (slc > th_high).astype('float32')
            lbl = (lbl == Labels.FREE_LEAD).astype('float32')
            slc_lead = slc_lead * (1-radii) + lbl * radii
            lead_labels.append(np.round(slc_lead).astype('bool'))
        # Convert to a proper numpy array
        lead_labels = np.asarray(lead_labels)
        # Remove any voxels that are background or grid
        lead_labels[labels==Labels.BACKGROUND] = False
        lead_labels[labels==Labels.GRID] = False
        del xx, yy
        return lead_labels
    
    def update_thresholds(self, arr):
        """Set *self.bg_threshold* and *self.lead_threshold* based on values
        in *arr*.
        
        """
        log.debug("Updating thresholds...")
        th_low, th_high = skimage.filters.threshold_multiotsu(arr)
        log.info("New thresholds: %f and %f", th_low, th_high)
        self.bg_threshold = th_low
        self.lead_threshold = th_high
    
    def update_profile(self, vol):
        """Set *self.lead_profile* with the number of voxels of lead in each
        slice.
        
        Be sure to call *update_thresholds* first, or this will not
        produce useful results.
        
        """
        log.debug("Updating lead profile...")
        labels = self.label_volume(vol)
        self.lead_profile = np.count_nonzero(labels>=Labels.FREE_LEAD, axis=(1,2))

    def profile_thresholds(self, profile, n_thresholds=3):
        # Determine if we need to reduce dynamic range
        dr_reduction = max(1, np.max(profile) / self.max_counts)
        log.debug("Reducing profile dynamic range by a factor of %f", dr_reduction)
        if dr_reduction > 1:
            profile = profile / dr_reduction
        # Determine the thresholds by Otsu's method
        ths = skimage.filters.threshold_multiotsu(profile, classes=n_thresholds)
        # Undo the dynamic range reduction
        if dr_reduction > 1:
            ths = [th * dr_reduction for th in ths]
            # th_low = th_low * dr_reduction
            # th_high = th_high * dr_reduction
        return ths
    
    def active_material_mask(self):
        log.debug("Calculating active material mask by thresholding.")
        if self.lead_profile is None:
            log.debug("No slice profile found, using all slices.")
            return ()
        # Determine lead thresholds for the given profile
        th_low, th_high = self.profile_thresholds(self.lead_profile)
        log.debug("Lead profile thresholds: %d and %d", th_low, th_high)
        mask = np.logical_and(self.lead_profile > th_low,
                              self.lead_profile < th_high)
        return mask
    
    def grid_mask(self):
        log.debug("Calculating grid mask by thresholding.")
        if self.lead_profile is None:
            msg = ("No lead slice profile found, unable to calculate grid mask."
                   "Try calling *update_profile()* first.")
            log.error(msg)
            raise exceptions.NoLeadProfile(msg)
        # Determine lead thresholds for the given profile
        th = self.profile_thresholds(self.lead_profile, 2)
        log.debug("Lead profile threshold: %d", th[0])
        mask = self.lead_profile > th
        return mask
    
    def label_volume(self, vol):
        """Create a volume with voxels of labels for each phase.
        
        Uses *self.bg_threshold* and *self.lead_threshold* to
        determine the different phases.
        
        Parameters
        ==========
        vol
          A numpy array with the density data to be labeled.
        
        Returns
        =======
        labels
          A numpy array with same shape as *vol* and "uint8" dtype,
          where each voxel's value describes which phase it is labeled
          as.
        
        """
        log.debug("Labeling %s volume", str(vol.shape))
        # Create array for labels, assume it's active material and set BG/lead
        labels = np.full_like(vol, Labels.ACTIVE_MATERIAL, dtype="uint8")
        # Set labels based on previously calculated thresholds
        labels[vol >= self.lead_threshold] = Labels.FREE_LEAD
        labels[vol < self.bg_threshold] = Labels.BACKGROUND
        return labels
    
