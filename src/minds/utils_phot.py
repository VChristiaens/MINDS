#! /usr/bin/env python

"""
Module with functions used to extract the aperture photometry from stage 3
MIRI/MRS spectral cubes, and recentering them. These are based on photutils and
VIP routines, respectively.

More information on VIP here: https://vip.readthedocs.io/
"""


__author__ = 'V. Christiaens'
__all__ = ['recenter_cubes',
           'extract_ap',
           'write_x1d',
           'fit_2dgaussian_bpm',
           'calc_scal_fac',
           'identify_scalefac_vs_bending',
           'bend_spectrum',
           'find_nearest']


from os.path import basename, isfile, splitext

import numpy as np
import pandas as pd
import photutils
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.convolution import interpolate_replace_nans as interp_nan
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.stats import (
    gaussian_fwhm_to_sigma,
    gaussian_sigma_to_fwhm,
    sigma_clip,
    sigma_clipped_stats,
)
from matplotlib import pyplot as plt
from packaging import version

if version.parse(photutils.__version__) >= version.parse('0.3'):
    # for photutils version >= '0.3' use photutils.centroids.centroid_com
    from photutils.centroids import centroid_com as cen_com
else:
    # for photutils version < '0.3' use photutils.centroid_com
    import photutils.centroid_com as cen_com
import pdb

from jwst import datamodels
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import (
    cube_collapse,
    cube_recenter_dft_upsampling,
    cube_shift,
    frame_crop,
)
from vip_hci.var import cube_filter_lowpass, frame_center, get_square, mask_circle


def recenter_cubes(filename, suffix='_cen', sig=3, method='cc',
                   imlib='ndimage-interp', crop_sz=9, verbose=True,
                   debug=False,  overwrite=False):
    """
    Make cube square and place star centroid on central pixel. Uses the
    following procedure:
        i) Pad input frames to squares;
        ii) Create binary masks from input cubes (with 1 for non-zero values);
        iii) NaN the zeros in the cube, interpolate them, and zero back NaNs;
        iv) Find coordinates of PSF in median frame;
        v) Estimate std of stellar coordinates over the cube;
        vi) Find shifts to align PSF in individual frames (prone to outliers);
        vii) Replace outlier shift values identified based on std value, using
        a linear interpolation.
        viii) Apply recentering shifts to cube (FFT method) and mask (bilinear)
        ix) Replace cube values to zero where binary mask < 0.5 after shift.

    Parameters
    ----------
    filename : str
        Filename for the input cube
    suffix : str
        Suffix to be appended to output filename.
    sig : float
        Number of standard deviations from the median coordinates to consider
        an estimated position as outlier.
    method: str, opt, {None, 'cc', 'gauss'}
        Method used for recentering: None for no recentering (only inferring
        median centroid position);'cc' for cross-correlation based re-alignment,
        'gauss' for 2D Gaussian fit based recentering.
    crop_sz: int, opt
        Crop size for sub-image in which a 2D Gaussian model is fit to find star
        centroid.
    imlib: str {'opencv', 'ndimage-interp', 'vip-fft'}, optional
        Method used for sub-pixel image shifts. 'opencv' to use a Lanczos4
        interpolation kernel (fastest), 'ndimage-interp' for a biquintic
        interpolation kernel, 'vip-fft' to use a FT-based method. The latter
        better preserves the total flux but is prone to Gibbs artefacts for
        poorly sampled images.
    verbose : bool
        Whether to print more information while processing.
    debug : bool
        Whether to show intermediate diagnostic plots, useful for debugging.
    overwrite : bool
        If True, overwrites output files.

    Returns
    -------
    None (the outputs are written as fits files)

    """

    # check if output files already exists
    bname = splitext(filename)[0]
    if bname[-5:] == '_bpc2':
        bname = bname[:-5]
    outname = bname+suffix+'.fits'
    if isfile(outname) and not overwrite:
        return None

    # load data
    cube = fits.getdata(filename, 'SCI')
    err = fits.getdata(filename, 'ERR')
    dqs = fits.getdata(filename, 'DQ')
    wm = fits.getdata(filename, 'WMAP')
    mask = np.zeros_like(dqs)
    mask[np.where(dqs == 0)] = 1

    DO_NOT_USE = 1
    bad_mask = []
    for dq in dqs:
        bad_mask.append(np.bitwise_and(
            dq.astype(int), DO_NOT_USE) == DO_NOT_USE)
    bad_mask = np.array(bad_mask)

    # Find frames full of NaNs
    bad_idx = np.all(~np.isfinite(cube.reshape(cube.shape[0], -1)),
                     axis=1)

    # pad to squares
    nz, ny, nx = cube.shape
    dxy0 = int(np.floor(((nx-ny))/2))
    dxy1 = int(np.ceil((nx-ny)/2))
    pad_x0 = max(int(-dxy0), 0)
    pad_x1 = max(int(-dxy1), 0)
    pad_y0 = max(int(dxy0), 0)
    pad_y1 = max(int(dxy1), 0)
    if nx != ny:
        cube = np.pad(cube, ((0, 0), (pad_y0, pad_y1), (pad_x0, pad_x1)))
        err = np.pad(err, ((0, 0), (pad_y0, pad_y1), (pad_x0, pad_x1)))
        mask = np.pad(mask, ((0, 0), (pad_y0, pad_y1), (pad_x0, pad_x1)))
        dqs = np.pad(dqs, ((0, 0), (pad_y0, pad_y1),
                     (pad_x0, pad_x1)), constant_values=513)
        wm = np.pad(wm, ((0, 0), (pad_y0, pad_y1), (pad_x0, pad_x1)))

    # NaN the zeros
    cube[np.where(cube == 0)] = np.nan
    err[np.where(err == 0)] = np.nan
    wm[np.where(wm == 0)] = np.nan
    # Interpolate the NaNs
    for z in range(nz):
        cube[z] = interp_nan(cube[z], Gaussian2DKernel(x_stddev=1, y_stddev=1),
                             convolve=convolve_fft)
        err[z] = interp_nan(err[z], Gaussian2DKernel(x_stddev=1, y_stddev=1),
                            convolve=convolve_fft)
        wm[z] = interp_nan(wm[z], Gaussian2DKernel(x_stddev=1, y_stddev=1),
                           convolve=convolve_fft)
    # Zero back the residual NaNs
    cube[np.where(np.isnan(cube))] = 0
    err[np.where(np.isnan(err))] = 0
    wm[np.where(np.isnan(wm))] = 0

    # fit_2dgaussian on collapsed image
    # compute weights for a weighted mean
    # cy_tmp, cx_tmp = frame_center(cube[0])

    cube_conv = cube_filter_lowpass(cube, fwhm_size=2)
    cy_tmp, cx_tmp = frame_center(cube_conv)

    final_cxy = np.zeros([nz, 2])
    fwhm_xyt = np.zeros([nz, 3])
    flags = np.ones([nz, 1])

    if method != 'gauss':
        if method == 'cc':
            res = cube_recenter_dft_upsampling(cube_conv, negative=False, fwhm=2,
                                               subi_size=None, upsample_factor=100,
                                               imlib='opencv', interpolation='lanczos4',
                                               mask=None, border_mode='reflect',
                                               full_output=True, verbose=verbose,
                                               nproc=1, save_shifts=False,
                                               debug=debug, plot=debug)
            cube_conv, final_cxy[:, 1], final_cxy[:, 0] = res
            final_cxy[:, 1] *= -1
            final_cxy[:, 0] *= -1
            fit_results = np.hstack([final_cxy, flags])
            fit_results = pd.DataFrame(fit_results, columns=['x', 'y', 'flags'])

        # in case of 'ddither' background subtraction, a lot of negatives can be present
        # safer to avoid them for the Gaussian fit
        cube_pos = cube_conv.copy()
        cube_pos[np.where(cube_pos < 0)] = np.nan
        # set NaN values to zero - NEW v6d
        # cube_pos[np.where(np.isnan(cube_pos))] = 0
        spec_proxy = np.zeros(nz)
        for z in range(nz):
            # interpolate the NaN values
            cube_pos[z] = interp_nan(cube_pos[z],
                                     Gaussian2DKernel(x_stddev=1, y_stddev=1),
                                     convolve=convolve_fft)
            mask_in = mask_circle(cube_pos[z], 6, mode='out', cy=cy_tmp,
                                  cx=cx_tmp) * mask[z]
            mask_out = mask_circle(cube_pos[z], 6, mode='in', cy=cy_tmp,
                                   cx=cx_tmp) * mask[z]
            mean_out = np.nanmean(mask_out[np.where(mask_out > 0)])
            std_out = np.nanstd(mask_out[np.where(mask_out > 0)])
            spec_proxy[z] = np.nansum(mask_in)/(mean_out+std_out)
            # spec_proxy[z] = np.abs(np.sum(mask_cube[z]))
        #cube_conv = cube_filter_lowpass(cube_pos, fwhm_size=2)
        # discard frames that are full of NaNs (can happen with latest JWST pipeline version)
        # 2d_arr = np.reshape(cube_conv, (cube_conv.shape[0], -1))
        # nnans = np.nansum(2d_arr, axis=1)
        good_idx = ~bad_idx
        cube_pos = cube_pos[good_idx]
        spec_proxy = spec_proxy[good_idx]
        spec_proxy[np.where(~np.isfinite(spec_proxy))] = 0

        med_img = cube_collapse(cube_pos, mode='wmean',
                                w=spec_proxy)  # np.median(cube, axis=0)
        # save and double check what the algo sees
        if debug:
            cube_pos = np.append(cube_pos, med_img[np.newaxis, :, :], axis=0)
            write_fits(bname + '_smooth_for_cen.fits', cube_pos)

        # subtract minimum in cropped image + find max near center.
        med_img_crop = frame_crop(med_img, crop_sz, cenxy=(cx_tmp, cy_tmp))
        Imin = max(np.amin(med_img_crop), 0)
        cy_c, cx_c = np.unravel_index(np.argmax(med_img_crop),
                                      med_img_crop.shape)
        cy_c += (med_img.shape[-2]-crop_sz)//2
        cx_c += (med_img.shape[-1]-crop_sz)//2

        res_fit = fit_2dgaussian_bpm(med_img-Imin,  # mask_circle(med_img, 10, mode='out', # cy=cy_tmp, cx=cx_tmp),
                                     cent=(cx_c, cy_c), crop=True,
                                     cropsize=crop_sz, full_output=True,
                                     debug=debug)
        cy_med = float(res_fit['centroid_y'])
        cx_med = float(res_fit['centroid_x'])
        final_cxy[:, 1] += cy_med
        final_cxy[:, 0] += cx_med
    else:
        mask_cube = mask_circle(cube, 10, mode='out') * mask
        spec_proxy = np.zeros(nz)
        for z in range(nz):
            spec_proxy[z] = np.abs(np.sum(mask_cube[z]))
        # np.median(cube, axis=0)
        med_img = cube_collapse(cube, mode='wmean', w=spec_proxy)
        res_fit = fit_2dgaussian_bpm(med_img, cent=(cx_tmp, cy_tmp), crop=False,
                                     full_output=True, debug=debug)
        cy_med = float(res_fit['centroid_y'])
        cx_med = float(res_fit['centroid_x'])
        for z in tqdm(range(nz)):
            debug_tmp = (z == 0) & debug
            dict_res = fit_2dgaussian_bpm(cube_conv[z], crop=True,
                                          cent=(cx_med, cy_med),
                                          bpm=bad_mask[z], cropsize=crop_sz,
                                          full_output=True, debug=debug_tmp)
            final_cxy[z, 0] = dict_res['centroid_x']
            final_cxy[z, 1] = dict_res['centroid_y']
            fwhm_xyt[z, 0] = dict_res['fwhm_x']
            fwhm_xyt[z, 1] = dict_res['fwhm_y']
            fwhm_xyt[z, 2] = dict_res['theta']

        fit_results = np.hstack([final_cxy, fwhm_xyt, flags])
        fit_results = pd.DataFrame(fit_results, columns=['x', 'y', 'fwhm_x',
                                                         'fwhm_y', 'theta',
                                                         'flags'])
        fit_results.to_csv(bname+'_2d_fit.csv', index=False)

    if method is None:
        cube_cen = cube
        err_cen = err
        dq_cen = dqs
        wm_cen = wm
    else:
        x_arr = sigma_clip(fit_results['x'].values, sigma=sig, cenfunc=np.nanmedian,
                           stdfunc=np.nanstd)
        y_arr = sigma_clip(fit_results['y'].values, sigma=sig, cenfunc=np.nanmedian,
                           stdfunc=np.nanstd)
        maskl = np.logical_or(x_arr.mask, y_arr.mask)

        if verbose:
            print(f"number of bad center positions: {np.sum(maskl)}")
        median_values = fit_results.median()

        fit_results.iloc[maskl, 0] = median_values[0]
        fit_results.iloc[maskl, 1] = median_values[1]
        if method == 'cc':
            fit_results.iloc[maskl, 2] = 0
        else:
            fit_results.iloc[maskl, 2] = median_values[2]
            fit_results.iloc[maskl, 3] = median_values[3]
            fit_results.iloc[maskl, 4] = median_values[4]
            fit_results.iloc[maskl, 4] = 0

        fit_results.to_csv(bname+'_2d_fit_robust.csv', index=False)

        # interpolate outliers
        idxr = np.arange(nz)
        cx_all = final_cxy[:, 0]
        cy_all = final_cxy[:, 1]
        interp_cy = interp1d(idxr[np.where(~maskl)], cy_all[np.where(~maskl)],
                             kind='linear', fill_value="extrapolate")
        interp_cx = interp1d(idxr[np.where(~maskl)], cx_all[np.where(~maskl)],
                             kind='linear', fill_value="extrapolate")
        cy_all[np.where(maskl)] = interp_cy(idxr[np.where(maskl)])
        cx_all[np.where(maskl)] = interp_cx(idxr[np.where(maskl)])
        final_cxy[:, 0] = cx_all
        final_cxy[:, 1] = cy_all

        # apply recentering shifts to cube and mask
        if imlib == 'ndimage-interp':
            interp = 'biquintic'
        else:
            interp = 'lanczos4'
        cube_cen = cube_shift(cube, cy_med-cy_all, cx_med-cx_all, imlib=imlib,
                              interpolation=interp)
        err_cen = cube_shift(err, cy_med-cy_all, cx_med-cx_all, imlib=imlib,
                             interpolation=interp)
        wm_cen = cube_shift(wm, cy_med-cy_all, cx_med-cx_all, imlib=imlib,
                            interpolation=interp)
        mask_cen = cube_shift(mask, cy_med-cy_all, cx_med-cx_all, imlib='opencv',
                              interpolation='bilinear')

        # The FFT-method can leave Gibbs artefacts which show up as high=spatial frequency
        # Smoothing with a fine Gaussian kernel deals with it while preserving flux.
        if imlib == 'vip-fft':
            cube_cen = cube_filter_lowpass(cube_cen, fwhm_size=1.5)
            err_cen = cube_filter_lowpass(err_cen, fwhm_size=1.5)
            wm_cen = cube_filter_lowpass(wm_cen, fwhm_size=1.5)

        # Replace cube values to zero where binary mask < 0.5
        cube_cen[np.where(mask_cen < 0.5)] = 0
        err_cen[np.where(mask_cen < 0.5)] = 0
        wm_cen[np.where(mask_cen < 0.5)] = 0
        mask_cen[np.where(mask_cen < 0.5)] = 0
        mask_cen[np.where(mask_cen >= 0.5)] = 1
        dq_cen = np.zeros_like(mask_cen)
        dq_cen[np.where(mask_cen == 0)] = 513

    # write output file
    with fits.open(filename) as hdul:
        # hdul = fits.open(filename)
        # Replace the original HDU with the recentered cube and DQ
        sci_hdu = hdul['SCI']
        sci_hdu.data = cube_cen.copy()
        err_hdu = hdul['ERR']
        err_hdu.data = err_cen.copy()
        dq_hdu = hdul['DQ']
        dq_hdu.data = dq_cen.copy()
        wm_hdu = hdul['WMAP']
        wm_hdu.data = wm_cen.copy()
        hdul.writeto(outname, overwrite=overwrite)

    if verbose:
        msg = "Centroid median xy pixel coordinates for {}: ({:.0f},{:.0f})"
        print(msg.format(basename(splitext(filename)[0]), cx_med, cy_med))
    return cx_med, cy_med


def extract_ap(fname, fname_x1d, suffix, cxy, fname_bkg=None, apcorr=None,
               apcorr_fn=None, list_aps=None, apsize=2.5, D=6.5, method='exact',
               subpixels=10, include_bkg_err=True, ann_removal=False,
               smooth_bkg=True, bkg_win=31, spike_filtering=False,
               sp_corr_win=15, sp_sig=5, max_nspax=2, neg_only=False,
               write_badflag=False, verbose=True, overwrite=False, debug=False):
    """
    Parameters
    ----------

    fname: str
        s3d file name
    fname_x1d: str
        x1d file name
    suffix: str
        Suffix to be appended to output filenames.
    cxy: tuple of 2 floats
        xy coordinates of the star centroid in each image of the cube.
    fname_bkg: str, opt
        [used if either include_bkg_err or add_diff_bkg are set to True]
        Filename (incl. full path) of associated BKG estimate propagated to
        level 3 (i.e. s3d).
    apcorr: None or 'FWHM', opt
        Whether the aperture size should be fixed for all channels (None) or
        proportional to the wavelength ('FWHM').
    apcorr_fn: str or None, opt
        Aperture correction filename. Should be provided if apcorr is True.
    apsize: float or int, opt
        Aperture size, expressed either in pixels or lambda over D (see apcorr).
    D: float, opt
        Effective diameter to consider for lambda/D calculation, in m.
    method: str {'no_spike', 'exact', 'subpixel'}, opt
        Method argument of photutils function. 'exact' is for exact flux
        considering fraction of each pixel, while 'subpixel' means rescaling
        the image first.
    subpixels: int, opt
        If method is 'subpixel', this is the resampling factor.
    include_bkg_err: bool, opt
        Whether to include background uncertainty to the flux and SB errors.
    ann_removal: bool, opt
        Whether to subtract the median pixel intensity beyond bkg_rad pixels
        from the star.
    smooth_bkg: bool, opt
        [ann_removal=True] Whether to smooth the background spectrum estimated
        from the annulus, before subtraction.
    bkg_win: int, opt
        [smooth_bkg = True] Window for the Savitzky-Golay filter used to smooth
        the background spectrum. An order 2 is used.
    spike_filtering: bool, opt
        Whether to filter spikes at the moment of aperture photometry.
    write_badflag: bool, opt
        Whether to write bad flags in a csv file.
    verbose : bool
        Whether to print more information while processing.
    overwrite : bool
        If True, overwrites output files.
    """

    fname_short = splitext(fname_x1d)[0][:-4]
    if isfile(fname_short+suffix+'.fits') and not overwrite:
        print("Spectrum already exists for {} - skipping extraction".format(fname_short))
        return None

    fname_short_bkg = fname_short.replace('psf', 'bkg')

    if verbose:
        print("Extracting spectrum for {}".format(fname_short))

    if include_bkg_err and fname_bkg is None:
        msg = "fname_bkg must be provided if include_bkg_err is set to True"
        raise ValueError(msg)

    if ann_removal:
        lab_ann = "_annulus"
    else:
        lab_ann = ''

    # Open relevant data cube
    cube = fits.getdata(fname, 'SCI')
    errcube = fits.getdata(fname, 'ERR')
    head = fits.getheader(fname, 'SCI')
    dqs = fits.getdata(fname, 'DQ')
    zp = np.zeros_like(dqs)
    zp[np.where(dqs == 0)] = 1
    weightmap = fits.getdata(fname, 'WMAP')
    nch = cube.shape[0]

    # Open bkg estimate if provided
    if fname_bkg is not None:
        bkg_cube = fits.getdata(fname_bkg, 'SCI')
        _, ny, nx = bkg_cube.shape
        # bkg_err_cube = fits.getdata(fname_bkg, 'ERR')
        bkg_dqs = fits.getdata(fname_bkg, 'DQ')
        bkg_zp = np.zeros_like(bkg_dqs)
        bkg_zp[np.where(bkg_dqs == 0)] = 1

    DO_NOT_USE = 1
    bad_mask = []
    for dq in dqs:
        bad_mask.append(np.bitwise_and(
            dq.astype(int), DO_NOT_USE) == DO_NOT_USE)
    bad_mask = np.array(bad_mask)

    pixar_sr = float(head['PIXAR_SR'])
    pxscale1 = float(head['CDELT1'])
    pxscale2 = float(head['CDELT2'])
    plsc = float(np.mean([pxscale1, pxscale2])*3600)

    # load wavelengths and define FWHM
    dm = datamodels.open(fname_x1d)
    wl = np.array(dm.spec[0].spec_table['WAVELENGTH'])
    # lod = wl*206265*1e-6/(D*plsc)  # commented, as previous definition not consistent with ap corr fac calculation
    fwhm = wl/plsc*0.31/8  # includes broadening factor

    # Identify non-zero frames in cube
    good_idx = np.where(np.nanmean(
        dqs.reshape(dqs.shape[0], -1), axis=1) != 513)[0]
    # good_cube = cube[good_idx].copy()

    if ann_removal:
        FWHM_low = (0.31/8.0 * 4.88)
        FWHM_high = (0.31/8.0 * 28.34)
        # same definition as in the notebook used to calculate aperture correction factors
        rIn_int = interp1d([4.88, 28.34], [FWHM_low*5, FWHM_high*3.0], 'linear',
                           fill_value='extrapolate')
        rOut_int = interp1d([4.88, 28.34], [FWHM_low*7.5, FWHM_high*3.75],
                            'linear', fill_value='extrapolate')
        rIn = rIn_int(wl)
        rOut = rOut_int(wl)

    final_sb = np.zeros([nch, 2])
    final_badflag = np.zeros([nch, 1])
    final_fluxes = np.zeros([nch, 2])
    npix = np.zeros(nch)
    flux = np.zeros(nch)
    flux_err = np.zeros(nch)
    bkg_flux = np.zeros(nch)
    bkg_sb = np.zeros(nch)
    # bbkg_lvls = np.zeros(nch)
    if include_bkg_err:
        final_bkg_ferr = np.zeros(nch)
        final_bkg_sberr = np.zeros(nch)

    if apcorr == 'FWHM':
        apsz = fwhm*apsize
        apcorr_fac = open_fits(apcorr_fn.format(lab_ann))
        apcorr_fac = apcorr_fac[list_aps.index(apsize)]
        wvl = apcorr_fac[0]
        corr = apcorr_fac[1]
        corr_int = interp1d(wvl, corr)
    else:
        apsz = np.ones(nch)*apsize

    if method != 'no_spike':
        method_p = method
    else:
        method_p = 'exact'

    # If requested, produce a spike corrected spectral cube,
    # in particular for spaxels that will be used in aperture photometry
    if spike_filtering:
        cube = spike_filter(cube, sp_corr_win, cxy, apsz, sig=sp_sig,
                            max_nspax=max_nspax, neg_only=neg_only, debug=debug)

    for z, ch in enumerate(good_idx):
        temp_weightmap = weightmap[ch, :, :]
        temp_weightmap[temp_weightmap > 1] = 1

        aper = photutils.CircularAperture(cxy, apsz[ch])

        phot_table = photutils.aperture_photometry(temp_weightmap, aper,
                                                   method=method_p,
                                                   subpixels=subpixels)
        aperture_area = float(phot_table['aperture_sum'][0])
        npix[ch] = aperture_area

        aper_phot = photutils.aperture_photometry(cube[ch], aper,
                                                  method=method_p,
                                                  subpixels=subpixels)
        flux[ch] = np.array(aper_phot['aperture_sum'])[0]

        aper_phot_err = photutils.aperture_photometry(errcube[ch], aper,
                                                      method=method_p,
                                                      subpixels=subpixels)
        flux_err[ch] = np.array(aper_phot_err['aperture_sum'])[0]

        if include_bkg_err:
            npx = int(np.sum(zp[ch]))
            nap = int(npx/npix[ch])
            if nap < 3:
                msg = "WARNING: Number of apertures less than 3!"
                msg += "BKG noise estimate may not be reliable."
                print(msg)
                # Commented to also run for 3.5 FWHM aperture:
                #raise ValueError("Number of apertures less than 3!")
            bkg_fluxes = np.zeros(nap)
            ap = 0
            c = 0
            for y in range(ny):
                for x in range(nx):
                    if bkg_zp[ch, y, x]:
                        bkg_fluxes[ap] += float(bkg_cube[ch, y, x])
                        c += 1
                    if c > npix[ch]:
                        ap += 1
                        c = 0
                    if ap > nap-1:
                        break
                if ap > nap-1:
                    break
            # correction for small sample stats
            bkg_err = np.std(bkg_fluxes)*np.sqrt(1+(1/nap))
            flux_err[ch] = np.sqrt(flux_err[ch]**2+bkg_err**2)
            bkg_flux[ch] = np.nanmean(bkg_fluxes)
            bkg_sb[ch] = bkg_flux[ch]/npix[ch]

        # if requested, remove median pixel intensity far from the star (bkg proxy)
        if ann_removal:
            # new
            annulus_aperture = photutils.CircularAnnulus(cxy, r_in=rIn[ch]/plsc,
                                                         r_out=rOut[ch]/plsc)
            annulus_stats = photutils.ApertureStats(cube[ch]*zp[ch],
                                                    annulus_aperture,
                                                    sum_method='exact')

            bkg_sb[ch] = annulus_stats.median
            bkg_flux[ch] = bkg_sb[ch] * npix[ch]

            # old
            # mask_img = mask_circle(cube[ch]*zp[ch], radius=bkg_rad, fillwith=0,
            #                        mode='in', cy=cxy[1], cx=cxy[0])
            # cube_err_tmp = errcube[ch]
            # bkg_lvls[ch] = np.average(mask_img[np.where(mask_img != 0)],
            #                           axis=None,
            #                           weights=1./np.power(cube_err_tmp[np.where(mask_img != 0)], 2))
            # flux_bkg[ch] = (bkg_lvls[ch]*npix[ch])

    if ann_removal:
        if smooth_bkg:
            # smooth
            bkg_flux = savgol_filter(bkg_flux, window_length=bkg_win,
                                     polyorder=2, mode='mirror')

    for z, ch in enumerate(good_idx):
        if ann_removal:
            flux[ch] -= bkg_flux[ch]
            if smooth_bkg:
                bkg_sb[ch] = bkg_flux[ch] / npix[ch]
        # if requested, add excess bkg estimate:
        # if add_diff_bkg:
        #     aper_phot = photutils.aperture_photometry(bkg_cube[ch], aper,
        #                                               method=method_p,
        #                                               subpixels=subpixels)
        #     bkg_flux = np.array(aper_phot['aperture_sum'])[0]

        #     mask_img = mask_circle(bkg_cube[ch]*bkg_zp[ch], radius=bkg_rad,
        #                            fillwith=0, mode='in', cy=cxy[1], cx=cxy[0])
        #     cube_err_tmp = bkg_err_cube[ch]
        #     bbkg_lvls[ch] = np.average(mask_img[np.where(mask_img != 0)],
        #                                axis=None,
        #                                weights=1./np.power(cube_err_tmp[np.where(mask_img != 0)], 2))
        #     exc_bkg_flux = bkg_flux-(bbkg_lvls[ch]*npix[ch])

        #     flux[ch] += exc_bkg_flux

        dq_phot = photutils.aperture_photometry(bad_mask[z].astype('float'),
                                                aper)
        if dq_phot['aperture_sum'] > 0.:
            final_badflag[ch, 0] = 1

        if apcorr == 'FWHM':
            final_corr = corr_int(wl[ch])
            flux[ch] *= final_corr
            flux_err[ch] *= final_corr
            if include_bkg_err:
                bkg_err *= final_corr
        # # multiply SB by Sr to have final fluxes
        final_sb[ch, 0] = flux[ch]/npix[ch]
        final_fluxes[ch, 0] = flux[ch]*pixar_sr * 1e6  # convert from MJy to Jy
        final_sb[ch, 1] = flux_err[ch]/npix[ch]
        final_fluxes[ch, 1] = flux_err[ch] * pixar_sr * 1e6  # MJy to Jy
        if include_bkg_err:
            final_bkg_sberr[ch] = bkg_err/npix[ch]
            final_bkg_ferr[ch] = bkg_err*pixar_sr*1e6

    del temp_weightmap
    final_badflag = final_badflag.astype('bool')

    # Open x1d file containing the spectrum of the source, update the spectrum but preserve header
    hdul = fits.open(fname_x1d)
    hdul.verify('ignore')
    table = hdul[1].data
    table['SURF_BRIGHT'] = final_sb[:, 0]
    table['SB_ERROR'] = final_sb[:, 1]
    table['FLUX'] = final_fluxes[:, 0]
    table['FLUX_ERROR'] = final_fluxes[:, 1]
    if include_bkg_err:
        table['FLUX_VAR_RNOISE'] = final_bkg_ferr
        table['SB_VAR_RNOISE'] = final_bkg_sberr
    hdul[1].data = table.copy()
    if write_badflag:
        badflag_df = pd.DataFrame(final_badflag[:, 0], columns=['BAD_FLAG'])
        badflag_df.to_csv(fname_short+suffix+'_badflag.csv', index=False)
    hdul.writeto(fname_short+suffix+'.fits', overwrite=overwrite)
    hdul.close()

    # save bkg spectrum
    if fname_bkg is not None or ann_removal:
        hdul = fits.open(fname_x1d)
        hdul.verify('ignore')
        table = hdul[1].data
        table['SURF_BRIGHT'] = bkg_sb
        table['FLUX'] = bkg_flux*pixar_sr*1e6
        if include_bkg_err:
            table['SB_ERROR'] = final_bkg_sberr
            table['FLUX_ERROR'] = final_bkg_ferr
        hdul.writeto(fname_short_bkg+suffix+'.fits', overwrite=overwrite)
        hdul.close()


def spike_filter(cube, sp_corr_win, cxy, apsz, sig=3, max_nspax=1,
                 neg_only=False, debug=False):
    # Principle:
    # for each point of each spaxel spectrum: fit sp_corr_win-1 surrounding neighbours with:
    # 1. linear trend
    # 2. quadratic trend
    # Evaluate stddev from the fit-data values
    # evaluate whether the point is discrepant by n stddev.

    def _outlier_corr(spec_obs, sg15, sg31, sig=3):
        spec_corr = spec_obs.copy()
        flags = np.zeros_like(spec_corr)
        #std = sigma_clipped_stats(spec_corr-sg31, sigma=5)[2]
        std = np.std(spec_corr-sg31)
        if neg_only:
            cond = (spec_corr-sg15 < -sig*std)
        else:
            cond = (np.abs(spec_corr-sg15) > sig*std)
        spec_corr[np.where(cond)] = sg15[np.where(cond)]
        flags[np.where(cond)] = 1

        return spec_corr, flags, std

    nz, ny, nx = cube.shape

    rad = int(np.amax(apsz))  # make sure to include all relevant spaxels
    mask = np.zeros([cube.shape[1], cube.shape[2]])
    mask = mask_circle(mask, rad, 1, cy=cxy[1], cx=cxy[0])
    nap = int(np.sum(mask))
    spax_corr = np.zeros([nap, cube.shape[0]])
    flags = np.zeros_like(spax_corr)

    c = 0
    #tmp_diff15 = spax_corr.copy()
    #tmp_diff31 = spax_corr.copy()
    for y in range(ny):
        for x in range(nx):
            # only correct where aperture photometry will be performed:
            if mask[y, x] > 0:
                sg31 = savgol_filter(cube[:, y, x], window_length=31,
                                     polyorder=2, mode='mirror')
                sg15 = savgol_filter(cube[:, y, x], window_length=15,
                                     polyorder=2, mode='mirror')
                res = _outlier_corr(cube[:, y, x], sg15, sg31, sig=sig)
                spax_corr[c, :], flags[c, :], std = res
                #tmp_diff15[c, :] = spax_corr[c, :]-sg15
                #tmp_diff31[c, :] = spax_corr[c, :]-sg31
                if debug:
                    plt.figure(figsize=(40, 20))
                    plt.plot(range(nz), cube[:, y, x], label='ori', color='k')
                    plt.plot(range(nz), sg15-sig*std,
                             label='sg2-{}sig'.format(sig), color='b')
                    plt.plot(range(nz), sg15+sig*std,
                             label='sg2+{}sig'.format(sig), color='y')
                    plt.plot(range(nz), spax_corr[c, :], label='corr',
                             color='r')
                    plt.legend()
                    plt.savefig("TMP_plot_spike_filter.pdf",
                                bbox_inches='tight')
                    plt.show()
                    pdb.set_trace()
                c += 1
    flag_sum = np.sum(flags, axis=0)
    # write_fits("TMP_diff31.fits", tmp_diff31)
    # write_fits("TMP_diff15.fits", tmp_diff15)
    # write_fits("TMP_flag_sum.fits", flag_sum)
    # pdb.set_trace()

    # final cube to be returned has spikes corrected only if present on max n spaxels
    c = 0
    cond1 = flag_sum > 0
    cond2 = flag_sum < max_nspax+1
    cond = cond1 & cond2
    for y in range(ny):
        for x in range(nx):
            # only correct where aperture photometry will be performed:
            if mask[y, x] > 0:
                cube[np.where(cond), y, x] = spax_corr[c, np.where(cond)]
                c += 1

    return cube


def write_x1d(out_fname, fname_x1d, wavelength, out_fluxes, overwrite=False):
    """
    Parameters
    ----------

    out_fname: str
        Output file name
    fname_x1d: str
        Original x1d file name
    wavelength: numpy 1d ndarray
        Wavelengths of the spectrum
    out_fluxes: numpy 1d ndarray
        Fluxes to be written
    overwrite : bool
        If True, overwrites output files.
    """
    # load wavelengths and compare
    dm = datamodels.open(fname_x1d)
    wl = np.array(dm.spec[0].spec_table['WAVELENGTH'])

    if not np.array_equal(wavelength, wl):
        idx_ori = find_nearest(wl, wavelength[0])
        idx_fin = find_nearest(wl, wavelength[-1])+1
    else:
        idx_ori = 0
        idx_fin = len(wl)

    # Open x1d file containing the spectrum of the source, update the spectrum but preserve header
    hdul = fits.open(fname_x1d)
    hdul.verify('ignore')
    table = hdul[1].data
    cols = hdul[1].columns
    ncols = cols.names
    ori_f = table['FLUX']

    # new x1d file
    ncol = len(cols)
    nc = []

    for i, c in enumerate(ncols):
        if ncol == 'SURF_BRIGHT':
            arr = table[c][idx_ori:idx_fin]*out_fluxes/ori_f[idx_ori:idx_fin]
        elif ncol == 'FLUX':
            arr = out_fluxes
        else:
            arr = table[c][idx_ori:idx_fin]
        nc.append(fits.Column(name=c, array=arr, format='D'))
    new_hdul1 = fits.BinTableHDU.from_columns(nc)
    new_hdul = fits.HDUList([hdul[0].copy(), new_hdul1, hdul[2].copy()])
    new_hdul.writeto(out_fname, overwrite=overwrite)
    hdul.close()
    new_hdul.close()


def fit_2dgaussian_bpm(array, cent, bpm=None, crop=False, cropsize=15, fwhmx=4,
                       fwhmy=4, theta=0, threshold=False, sigfactor=6,
                       full_output=True, debug=True):
    """ Fitting a 2D Gaussian to the 2D distribution of the data.

    Parameters
    ----------
    array : 2D numpy ndarray
        Input frame with a single PSF.
    cent : tuple of int, optional
        Approximate X,Y integer position of source in the array for extracting
        the subimage.
        If None the center of the frame is used for cropping the subframe (the
        PSF is assumed to be ~ at the center of the frame).
    bpm : 2D numpy ndarray, optional
        Bad pixel mask associated to input array.
    crop : bool, optional
        If True a square sub image will be cropped equal to cropsize.
    cropsize : int, optional
        Size of the subimage.
    fwhmx, fwhmy : float, optional
        Initial values for the standard deviation of the fitted Gaussian, in px.
    theta : float, optional
        Angle of inclination of the 2d Gaussian counting from the positive X
        axis.
    threshold : bool, optional
        If True the background pixels (estimated using sigma clipped statistics)
        will be replaced by small random Gaussian noise.
    sigfactor : int, optional
        The background pixels will be thresholded before fitting a 2d Gaussian
        to the data using sigma clipped statistics. All values smaller than
        (MEDIAN + sigfactor*STDDEV) will be replaced by small random Gaussian
        noise.
    full_output : bool, optional
        If False it returns just the centroid, if True also returns the
        FWHM in X and Y (in pixels), the amplitude and the rotation angle,
        and the uncertainties on each parameter.
    debug : bool, optional
        If True, the function prints out parameters of the fit and plots the
        data, model and residuals.

    Returns
    -------
    mean_y : float
        Source centroid y position on input array from fitting.
    mean_x : float
        Source centroid x position on input array from fitting.

    If ``full_output`` is True it returns a Pandas dataframe containing the
    following columns:
        'centroid_y': Y coordinate of the centroid.
        'centroid_x': X coordinate of the centroid.
        'fwhm_y': Float value. FHWM in X [px].
        'fwhm_x': Float value. FHWM in Y [px].
        'amplitude': Amplitude of the Gaussian.
        'theta': Float value. Rotation angle.
        # and fit uncertainties on the above values:
        'centroid_y_err'
        'centroid_x_err'
        'fwhm_y_err'
        'fwhm_x_err'
        'amplitude_err'
        'theta_err'

    """
    if array.ndim != 2:
        raise TypeError('Input array shoud  be 2D')

    if bpm is None:
        bpm = np.zeros_like(array).astype('bool')

    if crop:
        cenx, ceny = cent

        imside = array.shape[0]
        psf_subimage, suby, subx = get_square(array, min(cropsize, imside),
                                              ceny, cenx, position=True)
        bpm_subimage, _, _ = get_square(bpm, min(cropsize, imside),
                                        ceny, cenx, position=True)
    else:
        psf_subimage = array.copy()
        bpm_subimage = bpm.copy()

    if threshold:
        _, clipmed, clipstd = sigma_clipped_stats(psf_subimage, sigma=2)
        indi = np.where(psf_subimage <= clipmed + sigfactor * clipstd)
        subimnoise = np.random.randn(psf_subimage.shape[0],
                                     psf_subimage.shape[1]) * clipstd
        psf_subimage[indi] = subimnoise[indi]

    # Creating the 2D Gaussian model
    init_amplitude = np.ptp(psf_subimage[~bpm_subimage])
    xcom, ycom = cen_com(psf_subimage)
    gauss = models.Gaussian2D(amplitude=init_amplitude, theta=theta,
                              x_mean=xcom, y_mean=ycom,
                              x_stddev=fwhmx * gaussian_fwhm_to_sigma,
                              y_stddev=fwhmy * gaussian_fwhm_to_sigma)
    # Levenberg-Marquardt algorithm
    fitter = fitting.LevMarLSQFitter()
    y, x = np.indices(psf_subimage.shape)
    fit = fitter(gauss, x[~bpm_subimage], y[~bpm_subimage],
                 psf_subimage[~bpm_subimage])

    if crop:
        mean_y = fit.y_mean.value + suby
        mean_x = fit.x_mean.value + subx
    else:
        mean_y = fit.y_mean.value
        mean_x = fit.x_mean.value
    fwhm_y = fit.y_stddev.value*gaussian_sigma_to_fwhm
    fwhm_x = fit.x_stddev.value*gaussian_sigma_to_fwhm
    amplitude = fit.amplitude.value
    theta = np.rad2deg(fit.theta.value)

    # compute uncertainties
    if fitter.fit_info['param_cov'] is not None:
        with np.errstate(invalid='raise'):
            try:
                perr = np.sqrt(np.diag(fitter.fit_info['param_cov']))
                amplitude_e, mean_x_e, mean_y_e, fwhm_x_e, fwhm_y_e, theta_e = perr
                fwhm_x_e /= gaussian_fwhm_to_sigma
                fwhm_y_e /= gaussian_fwhm_to_sigma
            except:
                # this means the fit failed
                mean_y, mean_x = np.nan, np.nan
                fwhm_y, fwhm_x = np.nan, np.nan
                amplitude, theta = np.nan, np.nan
                mean_y_e, mean_x_e = np.nan, np.nan
                fwhm_y_e, fwhm_x_e = np.nan, np.nan
                amplitude_e, theta_e = np.nan, np.nan
    else:
        amplitude_e, theta_e, mean_x_e = np.nan, np.nan, np.nan
        mean_y_e, fwhm_x_e, fwhm_y_e = np.nan, np.nan, np.nan
        # the following also means the fit failed
        if fwhm_y == fwhmy and fwhm_x == fwhmx and amplitude == init_amplitude:
            mean_y, mean_x = np.nan, np.nan
            fwhm_y, fwhm_x = np.nan, np.nan
            amplitude, theta = np.nan, np.nan

    if debug:
        images = (psf_subimage, fit(x, y), psf_subimage-fit(x, y))
        vmin = np.amin(fit(x, y))
        vmax = np.amax(fit(x, y))
        if threshold:
            label = ('Subimage thresholded', 'Model', 'Residuals')
        else:
            label = ('Subimage', 'Model', 'Residuals')
        default_cmap = 'viridis'
        fig = plt.figure(figsize=(6, 2), dpi=300)
        for v in range(1, 4):
            ax = plt.subplot(1, 3, v)
            ax.set_aspect('auto')
            ax.imshow(images[v-1], cmap=default_cmap, origin='lower',
                      interpolation='nearest', vmin=vmin,
                      vmax=vmax)

            if label[v-1] is not None:
                ax.annotate(label[v-1], xy=(5, 5), color='w',
                            xycoords='axes pixels', weight='bold',  # size=label_size
                            )
            # plt.savefig(save, dpi=dpi, bbox_inches='tight', pad_inches=0,
            #            transparent=transparent)
            plt.show()
        # plot_frames((psf_subimage, fit(x, y), psf_subimage-fit(x, y)),
        #              grid=True, grid_spacing=1, label=label)
        print('FWHM_y =', fwhm_y)
        print('FWHM_x =', fwhm_x, '\n')
        print('centroid y =', mean_y)
        print('centroid x =', mean_x)
        print('centroid y subim =', fit.y_mean.value)
        print('centroid x subim =', fit.x_mean.value, '\n')
        print('amplitude =', amplitude)
        print('theta =', theta)

    if full_output:
        return {'centroid_y': mean_y, 'centroid_x': mean_x, 'fwhm_y': fwhm_y,
                'fwhm_x': fwhm_x, 'amplitude': amplitude, 'theta': theta,
                'centroid_y_err': mean_y_e, 'centroid_x_err': mean_x_e,
                'fwhm_y_err': fwhm_y_e, 'fwhm_x_err': fwhm_x_e,
                'amplitude_err': amplitude_e, 'theta_err': theta_e}
    else:
        return mean_y, mean_x


def calc_scal_fac(w1, w2, f1, f2):
    idx_1 = find_nearest(w1, w2[0], constraint='ceil')+1
    idx_2 = find_nearest(w2, w1[-1], constraint='ceil')+1
    intf2 = interp1d(w2[:idx_2], f2[:idx_2])
    f2_int1 = intf2(w1[idx_1:])
    scal_facs = f1[idx_1:]/f2_int1
    scal_fac = np.mean(scal_facs)
    return scal_fac


def identify_scalefac_vs_bending(all_scale_facs, thr=0.05):
    # thr is the expected median relative accuracy for the photometric calibration (i.e. 0.03 means you expect most bands to be photometrically calibrated within ~3%)
    # nscal = len(all_scale_facs)
    final_scale_facs = [1]
    do_bend = [False]
    for s, sca in enumerate(all_scale_facs[1:-1]):
        if abs(sca-1) > thr and s < len(all_scale_facs[1:-1])-1 and abs(np.median(all_scale_facs[s+2:])-sca) > abs(np.median(all_scale_facs[s+2:])-1):
            final_scale_facs.append(sca)
            do_bend.append(True)
        elif not do_bend[s]:
            final_scale_facs.append(sca*final_scale_facs[s])
            do_bend.append(False)
        else:
            final_scale_facs.append(sca)
            do_bend.append(False)
    # add last one (can only be scaling)
    final_scale_facs.append(all_scale_facs[-1]*final_scale_facs[-1])
    do_bend.append(False)

    return final_scale_facs, do_bend


def bend_spectrum(scal_ori, scal_end, wl):
    # use linear interpolation to bend spectrum between 2 values
    bend_facs = np.linspace(scal_ori, scal_end, len(wl))
    return bend_facs


def find_nearest(array, value, output='index', constraint=None, n=1):
    """
    Function to find the indices, and optionally the values, of an array's n
    closest elements to a certain value.

    Parameters
    ----------
    array: 1d numpy array or list
        Array in which to check the closest element to value.
    value: float
        Value for which the algorithm searches for the n closest elements in
        the array.
    output: str, opt {'index','value','both' }
        Set what is returned
    constraint: str, opt {None, 'ceil', 'floor'}
        If not None, will check for the closest element larger than value (ceil)
        or closest element smaller than value (floor).
    n: int, opt
        Number of elements to be returned, sorted by proximity to the values.
        Default: only the closest value is returned.

    Returns
    -------
    Either:
        (output='index'): index/indices of the closest n value(s) in the array;
        (output='value'): the closest n value(s) in the array,
        (output='both'): closest value(s) and index/-ices, respectively.
    By default, only returns the index/indices.

    Possible constraints: 'ceil', 'floor', None ("ceil" will return the closest
    element with a value greater than 'value', "floor" the opposite)
    """
    if isinstance(array, np.ndarray):
        pass
    elif isinstance(array, list):
        array = np.array(array)
    else:
        raise ValueError("Input type for array should be np.ndarray or list.")

    if constraint is None:
        fm = np.absolute(array-value)
        idx = fm.argsort()[:n]
    elif constraint == 'floor' or constraint == 'ceil':
        indices = np.arange(len(array), dtype=np.int32)
        if constraint == 'floor':
            fm = -(array-value)
        else:
            fm = array-value
        crop_indices = indices[np.where(fm > 0)]
        fm = fm[np.where(fm > 0)]
        idx = fm.argsort()[:n]
        idx = crop_indices[idx]
        if len(idx) == 0:
            msg = "No indices match the constraint ({} w.r.t {:.2f})"
            print(msg.format(constraint, value))
            raise ValueError("No indices match the constraint")
    else:
        raise ValueError("Constraint not recognised")

    if n == 1:
        idx = idx[0]

    if output == 'index':
        return idx
    elif output == 'value':
        return array[idx]
    else:
        return array[idx], idx
