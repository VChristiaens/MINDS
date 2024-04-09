#! /usr/bin/env python

"""
Module with helper functions for pair-wise dither subtraction or dedicated BKG
subtraction on detector images.

More information here: https://vip.readthedocs.io/
"""


from vip_hci.fits import open_fits, write_fits
from vip_hci.var import frame_filter_highpass

__author__ = "V. Christiaens, M. Samland"
__all__ = [
    "read_file_info",
    "read_file_info2",
    "clean_background_subtraction",
    "detector_background_subtraction",
]

import os
import pdb
import warnings
from os.path import basename, dirname, isfile, join, splitext

import jwst
import numpy as np
import numpy.ma as ma
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from packaging import version
from scipy.optimize import minimize

vjwst = jwst.__version__
# print("JWST pipeline version: ", vjwst)

if version.parse(vjwst) < version.parse("1.8.0"):
    raise ValueError(
        "Please update JWST package to a version larger or equal than 1.8.0"
    )
# from matplotlib import pyplot as plt
if version.parse(vjwst) < version.parse("1.10.0"):
    from jwst.datamodels import dqflags
else:
    from stdatamodels.jwst.datamodels import dqflags

# from utils_bpc_006 import det_corr_line

# from jwst import datamodels


def read_file_info(files, columns):
    info_table = []
    for file in files:
        hdr = fits.getheader(file)
        file_info = [hdr[x] for x in columns]
        file_info.append(file)
        info_table.append(file_info)
    column_names = columns + ["FILE_PATH"]
    info_table = pd.DataFrame(info_table, columns=column_names)
    return info_table


def read_file_info2(files, columns, columns2):
    info_table = []
    for file in files:
        hdr = fits.getheader(file)
        hdul = fits.open(file)
        hdr2 = hdul[1].header
        file_info = [hdr[x] for x in columns]
        file_info2 = [hdr2[x] for x in columns2]
        file_info.extend(file_info2)
        file_info.append(file)
        info_table.append(file_info)
    column_names = columns + columns2 + ["FILE_PATH"]
    info_table = pd.DataFrame(info_table, columns=column_names)
    return info_table


def detector_background_subtraction(
    info_table,
    reference_info_table=None,
    nodding=True,
    suffix="_bgsub",
    overwrite=False,
):
    """Performs background subtraction by directly subtracting two files from each other.
       The appropriate files to subtract are picked automatically and data quality flags
       are set properly in the resulting new file. The new files will have the ending
       "_bgsub.fits".

    Parameters
    ----------
    info_table : table
        Table containing important header information relevant for reference selection
        of science files.
    reference_info_table : table
        Table containing important header information for pool of background reference
        files. If None, the reference files are assumed to come from the same data set
        as the science data (e.g., nodding-subtraction).
    nodding : Boolean
        If True, only reference backgrounds from different dither position are chosen.
    suffix : String
        Suffix to be added to end of file name for output.
    overwrite : Boolean
        If True, overwrites output files.

    Returns
    -------
    type
        Description of returned object.

    """

    if reference_info_table is None:
        reference_info_table = info_table

    # determine dith_association based on header value for NDITHER
    hdul = fits.open(info_table["FILE_PATH"].iloc[0])
    hdr = hdul[0].header
    ndith = int(hdr['NUMDTHPT'])
    if ndith == 2:
        dith_association = "A"
    elif ndith == 4:
        dith_association = "B"
    else:
        msg = "Your observations were taken with neither 2 nor 4 dither points."
        msg += " You can only use this notebook with bkg_method = 'annulus'."
        raise ValueError(msg)

    for i in range(len(info_table)):

        channel_target, band_target, patt_num = info_table.iloc[i][
            ["CHANNEL", "BAND", "PATT_NUM"]
        ]

        if dith_association == "A":
            if patt_num == 1:
                ref_patt_num = 2
            elif patt_num == 2:
                ref_patt_num = 1
            elif patt_num == 3:
                ref_patt_num = 4
            elif patt_num == 4:
                ref_patt_num = 3
        else:
            if patt_num == 1:
                ref_patt_num = 4
            elif patt_num == 2:
                ref_patt_num = 3
            elif patt_num == 3:
                ref_patt_num = 2
            elif patt_num == 4:
                ref_patt_num = 1

        if nodding:
            mask = np.logical_and.reduce(
                [
                    reference_info_table["CHANNEL"] == channel_target,
                    reference_info_table["BAND"] == band_target,
                    reference_info_table["PATT_NUM"] == ref_patt_num,
                ]
            )
        else:
            mask = np.logical_and(
                reference_info_table["CHANNEL"] == channel_target,
                reference_info_table["BAND"] == band_target,
            )

        reference_files = reference_info_table[mask]["FILE_PATH"].values

        bg_cube = []
        dq_cube = []
        for ref_file in reference_files:
            bg_cube.append(fits.getdata(ref_file, "SCI"))
            dq_cube.append(fits.getdata(ref_file, "DQ"))
        bg_cube = np.array(bg_cube)
        dq_cube = np.array(dq_cube)

        # datamodels.dqflags.pixel
        DO_NOT_USE = 1  # datamodels.dqflags.pixel['DO_NOT_USE']
        if len(bg_cube) > 1:
            bad_mask = []
            for dq in dq_cube:
                bad_mask.append(np.bitwise_and(dq, DO_NOT_USE) == DO_NOT_USE)
            bad_mask = np.array(bad_mask)
            bg_masked_arr = ma.masked_array(bg_cube, mask=bad_mask)
            bg_image_median = ma.median(bg_masked_arr, axis=0)
            bg_image = bg_image_median.data
            # Only flags DO_NOT_USE for pixels that are marked as such in all frames
            dq_image = bg_image_median.mask.astype(np.uint32)
        else:
            bg_image = bg_cube[0]
            dq_image = dq_cube[0]

        sci_filename = info_table["FILE_PATH"].iloc[i]
        dirname = os.path.dirname(sci_filename)
        basename, ext = os.path.splitext(os.path.basename(sci_filename))
        outname = os.path.join(dirname, basename + suffix + ext)

        print(f"Background subtracting: {sci_filename}")

        if not isfile(outname) or overwrite:
            with fits.open(sci_filename) as hdul:
                hdul = fits.open(sci_filename)
                sci_hdu = hdul["SCI"]
                # Replace the original HDU with the new one
                sci_hdu.data = sci_hdu.data - bg_image
                # Switch on DO_NOT_USE flag in DQ for bad pixels in background reference
                dq_hdu = hdul["DQ"]
                dq_hdu.data = np.bitwise_or(dq_hdu.data, dq_image)
                hdul.writeto(outname, overwrite=overwrite)


def clean_background_subtraction(
    info_table,
    dedicated_bg=False,
    suffix=["_psf", "_bkg"],
    sig=5,
    perc=50,
    kernel_sz=(7, 81),
    rm_persist=True,
    persist_frac=0.01,
    preserve_inter_slice=False,
    inter_slice_dq=512,
    verbose=True,
    debug=False,
    overwrite=False,
):
    """
    Performs a clean background subtraction on dither DETECTOR IMAGES (without
    previous pair-wise subtraction) by:
        i) identifying bad pixels (identical over 4 dithers);
        ii) removing persistence effects;
        iii) estimating the background (minimum over 4 dithers);
        iv) (optionally over several iterations) refining the background
        estimation by replacing significant outliers (that are not bad pixels).

    Parameters
    ----------
    info_table : table
        Table containing important header information relevant for pair
        associations (non-BKG subtracted cubes).
    dedicated_bg : bool, opt
        Whether dedicated background observations are available. If so, these
        will be used during stage 3 of the pipeline (where outliers in BKG
        images are also estimated and their effect removed).
    suffix : tuple or list of 2 strings, optional
        Suffixes to be added to end of file names for output PSF and BKG
        estimates, respectively.
    sig : float, optional
        Number of standard deviations above or below the median to consider
        outliers in BKG map estimate, after subtraction of estimated PSF.
        Outliers are interpolated with a Gaussian kernel.
    perc : float, optional
        In each moving filter cell, percentile of pixel intensities used as
        background proxy.
    kernel_sz: tuple of 2 floats or tuple of 4 tuples of 2 floats, optional
        Size in pixels (along y and x, respectively) of the 2D median filter
        kernel used to identify residual PSF signals in original BKG map.
    dq_thr: int, opt
        Value in DQ maps to consider as threshold for original bad pixel map.
    rm_persist : bool, optional
        Whether to remove persistence effects.
    persist_frac : float, optional
        Initial guess on the fraction of flux in the persistence artefact.
    preserve_inter_slice: bool, optional
        Whether to preserve values between slices. Note: irrelevant when done
        before spec2.
    verbose : bool, optional
        Whether to print more information while processing.
    debug : bool, optional
        Whether to show diagnostic plots for first channel of each cube.
    overwrite : bool, optional
        If True, overwrites output files.

    Returns
    -------
    None (the outputs are written as fits files)

    """

    def _band_name_to_int(bname):
        if bname == "LONG":
            return 2
        elif bname == "MEDIUM":
            return 1
        else:
            return 0

    def _ch_name_to_int(chname):
        if chname == "CH12":
            return 0
        else:
            return 1

    DO_NOT_USE = dqflags.pixel["DO_NOT_USE"]

    # adapt kernel_sz depending on whether a single tuple or 4 tuples are provided
    if len(kernel_sz) == 2 and not isinstance(kernel_sz[0], tuple):
        kernel_sz = (
            ((kernel_sz, kernel_sz), (kernel_sz, kernel_sz), (kernel_sz, kernel_sz)),
            ((kernel_sz, kernel_sz), (kernel_sz, kernel_sz), (kernel_sz, kernel_sz)),
        )
    elif len(kernel_sz) != 2 and len(kernel_sz[0]) != 3:
        msg = "Kernel size should be either (i) a tuple of 2 floats, (ii) a "
        msg += "tuple of 2 tuples of 2 floats, or a tuple of 2 tuples of "
        msg += "3 pairs of tuples, each with 2 floats (one per channel pair and band)."
        raise ValueError(msg)

    # CORRECTION OF PERSISTENCE EFFECTS
    if rm_persist:
        lab = "_rmp"
        for i in range(len(info_table)):

            sci_filename = info_table["FILE_PATH"].iloc[i]
            if dedicated_bg:
                dname = dirname(sci_filename)
                fname, ext = splitext(basename(sci_filename))
                outname_psf1 = join(dname, fname + suffix[0] + ext)
            else:
                outname_psf1 = splitext(sci_filename)[0] + lab + ".fits"

            if not isfile(outname_psf1) or overwrite:

                channel_target, band_target, patt_num = info_table.iloc[i][
                    ["CHANNEL", "BAND", "PATT_NUM"]
                ]

                # load current cube, and make intersected DQ
                psf_cube_curr = fits.getdata(sci_filename, "SCI")
                dq_curr = fits.getdata(sci_filename, "DQ")
                zp_curr = np.zeros_like(dq_curr)
                zp_curr[np.where(dq_curr == 0)] = 1

                # remove contribution from previous cube (if not first cube)
                if patt_num > 1:
                    if verbose:
                        print(
                            f"Removal of persistence effects for: {sci_filename}")
                    # previous cube
                    mask_prev = np.logical_and.reduce(
                        [
                            info_table["CHANNEL"] == channel_target,
                            info_table["BAND"] == band_target,
                            info_table["PATT_NUM"] == patt_num - 1,
                        ]
                    )
                    prev_file = info_table[mask_prev]["FILE_PATH"].values[0]

                    # load previous cube and dq
                    psf_cube_prev = fits.getdata(prev_file, "SCI")
                    dq_prev = fits.getdata(prev_file, "DQ")
                    zp_prev = np.zeros_like(dq_prev)
                    zp_prev[np.where(dq_prev == 0)] = 1

                    # intersecting dqs
                    inter_dq = zp_prev * zp_curr

                    # subtract persistence effect from original PSF
                    psf_cube_curr -= persist_frac * psf_cube_prev * inter_dq

                # write fits file with same format
                with fits.open(sci_filename) as hdul:
                    sci_hdu = hdul["SCI"]
                    # Replace the original HDU with the PSF one, and write
                    sci_hdu.data = psf_cube_curr.copy()
                    hdul.writeto(outname_psf1, overwrite=overwrite)
    else:
        lab = ""

    if not dedicated_bg:
        # detector_background_subtraction(info_table, reference_info_table=None,
        #                                 nodding=True, suffix='_bgsub',
        #                                 dith_association='B', overwrite=False)
        # CLEAN BKG SUBTRACTION from dithers
        n_sci = len(info_table)
        for i in range(n_sci):
            j = i + 1
            sci_filename = info_table["FILE_PATH"].iloc[i]

            if verbose:
                print(
                    f"Background estimation and subtraction for: {sci_filename} ({j}/{n_sci})"
                )
            frame_sci = fits.getdata(splitext(sci_filename)[
                                     0] + lab + ".fits", "SCI")
            channel_target, band_target, patt_num = info_table.iloc[i][
                ["CHANNEL", "BAND", "PATT_NUM"]
            ]

            inpath = dirname(sci_filename)

            bkg_tmp = "CH{}_B{}_bkg_fin.fits".format(
                channel_target, band_target)

            if not isfile(join(inpath, bkg_tmp)) or overwrite:
                # 1. Estimate common BKG from min intensities for each set of 4 dithers
                mask_same_CHB = np.logical_and.reduce(
                    [
                        info_table["CHANNEL"] == channel_target,
                        info_table["BAND"] == band_target,
                    ]
                )
                dith_files = info_table[mask_same_CHB]["FILE_PATH"].values

                ndith = len(dith_files)
                if ndith != 4 and ndith != 2:
                    raise ValueError(
                        "Only {} dither files found.".format(ndith))

                all_diths = np.empty(
                    [ndith, frame_sci.shape[0], frame_sci.shape[1]])
                all_dqs = np.zeros_like(all_diths)
                for nd, df in enumerate(dith_files):
                    all_diths[nd] = fits.getdata(df, "SCI")
                    all_dqs[nd] = fits.getdata(df, "DQ")

                bkg0 = np.amin(all_diths, axis=0)
                dq_min = np.amin(all_dqs, axis=0)
                dq_min = dq_min.astype(np.uint32)
                bkg0_hpf = frame_filter_highpass(
                    bkg0, mode="median-subt", median_size=3, conv_mode="conv"
                )
                bkg0_hpf[np.where(bkg0_hpf < 0)] = 0
                if debug:
                    write_fits(join(inpath, "all_diths.fits"),
                               all_diths, verbose=False)
                    write_fits(
                        join(
                            inpath,
                            "CH{}_B{}_bkg1.fits".format(
                                channel_target, band_target),
                        ),
                        bkg0,
                        verbose=False,
                    )

                # find outlier residual star lines
                bad_lines, _ = detect_outlier_lines(bkg0, dq_min, sig=sig)

                # dq_min[np.where(bad_lines)] += dq_thr+1
                dq_bl = np.zeros_like(dq_min)
                dq_bl[np.where(bad_lines)] = DO_NOT_USE
                dq_min = np.bitwise_or(dq_min, dq_bl)

                # set final BKG as NaN-median filter, after setting to NaN all bad values.
                ch = _ch_name_to_int(channel_target)
                bn = _band_name_to_int(band_target)
                bkg_fin = nanperc_filter(
                    bkg0, dq_min, perc=perc, kernel_sz=kernel_sz[ch][bn]
                )

                if debug:
                    bkg_tmp1 = "CH{}_B{}_bkg_NaN.fits".format(
                        channel_target, band_target
                    )
                    bkg_nan = bkg0.copy()
                    bkg_nan[np.where(bad_lines)] = np.nan
                    write_fits(join(inpath, bkg_tmp1), bkg_nan, verbose=debug)
                    write_fits(join(inpath, bkg_tmp), bkg_fin, verbose=debug)

            else:
                bkg_fin = open_fits(join(inpath, bkg_tmp))

            # set inter-slice values back to original
            if preserve_inter_slice:
                max_dq_bit = bin(int(np.amax(dq_min)))[2:]
                ndig_max = len(max_dq_bit)
                dq_bit = np.zeros([ndig_max, dq_min.shape[0], dq_min.shape[1]])
                for i in range(dq_min.shape[0]):
                    for j in range(dq_min.shape[1]):
                        dq_tmp = np.unpackbits(
                            np.array([int(dq_min[i, j])],
                                     dtype=">i8").view(np.uint8)
                        )
                        dq_bit[:, i, j] = dq_tmp[-ndig_max:]
                #        dq_bit = np.unpackbits(dq_tmp.view(np.uint8), axis=0)
                is_dq_bit = bin(inter_slice_dq)[2:]
                dig_is = len(is_dq_bit)
                bkg_fin[np.where(dq_bit[-dig_is] == 1)] = bkg0[
                    np.where(dq_bit[-dig_is] == 1)
                ]

            # WRITE FINAL PSF and BKG cubes
            psf = frame_sci - bkg_fin

            dname = dirname(sci_filename)
            fname, ext = splitext(basename(sci_filename))
            outname_psf2 = join(dname, fname + suffix[0] + ext)
            outname_bkg = join(dname, fname + suffix[1] + ext)

            with fits.open(sci_filename) as hdul:
                hdul = fits.open(sci_filename)
                sci_hdu = hdul["SCI"]
                # Replace the original HDU with the PSF one, and write
                sci_hdu.data = psf.copy()
                if not isfile(outname_psf2) or overwrite:
                    hdul.writeto(outname_psf2, overwrite=overwrite)
                # Replace the original HDU with the BKG one, and write
                sci_hdu.data = bkg_fin.copy()
                if not isfile(outname_bkg) or overwrite:
                    hdul.writeto(outname_bkg, overwrite=overwrite)

    else:
        msg = "Use bkg_method='ddither' for dedicated BKG subtraction."
        raise ValueError(msg)


def nanperc_filter(img, dq, perc=30, kernel_sz=(5, 81)):
    """
    Same as Nan-median filter, but nan-percentile filter.
    """

    DO_NOT_USE = dqflags.pixel["DO_NOT_USE"]

    try:
        if not isinstance(kernel_sz[0], tuple):
            kernel_sz = (kernel_sz, kernel_sz)

        hker_sz_l = (int((kernel_sz[0][0] - 1) / 2),
                     int((kernel_sz[0][1] - 1) / 2))
        hker_sz_r = (int((kernel_sz[1][0] - 1) / 2),
                     int((kernel_sz[1][1] - 1) / 2))
    except:
        pdb.set_trace()
    img_nan = img.copy()
    bp_map = np.bitwise_and(dq, DO_NOT_USE) == DO_NOT_USE
    img_nan[np.where(bp_map)] = np.nan

    filtered_img = np.zeros_like(img)
    ny, nx = img.shape
    nxm = nx / 2

    for y in range(ny):
        for x in range(nx):
            if x < nxm:
                hker_sz = hker_sz_l
            else:
                hker_sz = hker_sz_r
            y0 = max(0, y - hker_sz[0])
            yN = min(ny - 1, y + hker_sz[0]) + 1
            x0 = max(0, x - hker_sz[1])
            xN = min(nx - 1, x + hker_sz[1]) + 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                filtered_img[y, x] = np.nanpercentile(
                    img_nan[y0:yN, x0:xN], q=perc)

    filtered_img[np.where(np.isnan(filtered_img))] = 0

    return filtered_img


def detect_outlier_lines(bkg, dq, sig=5):
    ny = bkg.shape[0]
    nx = bkg.shape[1]
    nx2 = int(nx / 2)
    bkg_mspec = np.zeros([2, ny])
    bkg_sspec = np.zeros([2, ny])
    bad_lines = np.zeros_like(bkg)

    # discard non-finite values
    bpmap_ori = np.zeros_like(bkg).astype("bool")
    bpmap_ori[np.where(~np.isfinite(bkg))] = True
    bpmap_ori[np.where(bkg == 0.0)] = True

    # set pixels with non-finite values or zero as DO_NOT_USE (in case this is not already the case)
    #         dq_image = bpmap_ori.astype(np.uint32)
    #         ndq =  np.bitwise_or(hdul[3].data, dq_image)
    #         hdul[3].data = ndq

    DO_NOT_USE = dqflags.pixel["DO_NOT_USE"]
    # NON_SCIENCE = dqflags.pixel['NON_SCIENCE']

    mask_bad = np.bitwise_and(dq, DO_NOT_USE) == DO_NOT_USE
    # mask_non_science = np.bitwise_and(dq, NON_SCIENCE) == NON_SCIENCE

    gp_map = ~mask_bad

    # bpmap_ori = (mask_bad & ~mask_non_science) | (
    #     bpmap_ori & mask_non_science)

    # bpmap_ori = bpmap_ori.astype('int')

    for i in range(2):
        if i == 0:
            x0 = 0
            xN = nx2
        else:
            x0 = nx2
            xN = nx
        for j in range(ny):
            row = bkg[j, x0:xN]
            # dqr = dq[j, x0:xN]
            gpr = gp_map[j, x0:xN]
            tmp = row[np.where(gpr == 1)]
            if len(tmp) > 5:
                _, bkg_mspec[i, j], bkg_sspec[i, j] = sigma_clipped_stats(
                    tmp, sigma=2.5
                )
                cond1 = gpr == 1
                cond2 = row > bkg_mspec[i, j] + sig * bkg_sspec[i, j]
                bad_tmp = np.zeros_like(bad_lines[j, x0:xN])
                bad_tmp[np.where(cond1 & cond2)] = 1
                bad_lines[j, x0:xN] = bad_tmp.copy()

    return bad_lines, bkg_mspec
