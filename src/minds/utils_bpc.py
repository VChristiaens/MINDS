#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 13, 2022
Last modified on May 5, 2023 (by Matthias)

@author: Valentin Christiaens
"""

from os.path import basename, dirname, isfile, join, splitext

import jwst
import numpy as np
import pandas as pd
import vip_hci
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.convolution import interpolate_replace_nans as interp_nan
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

#from scipy.ndimage.filters import correlate
from packaging import version
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import (
    approx_stellar_position,
    cube_fix_badpix_clump,
    cube_fix_badpix_interp,
    frame_fix_badpix_isolated,
)
from vip_hci.var import frame_filter_highpass, mask_circle

vvip = vip_hci.__version__
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


def read_file_info(files, columns):
    info_table = []
    for file in files:
        hdr = fits.getheader(file)
        file_info = [hdr[x] for x in columns]
        file_info.append(file)
        info_table.append(file_info)
    column_names = columns + ['FILE_PATH']
    info_table = pd.DataFrame(info_table, columns=column_names)
    return info_table


def det_corr_cross(frame_corr, hpf_frame, bp_shape='+', sig=5, debug=False):

    ny, nx = frame_corr.shape

    # define cross shape for match-filtering of bad pixels
    if bp_shape == '+':
        szc = 7  # NOTE: not bad with 5 although some false positives from cross-shaped residual lines
        mszc = int((szc-1)/2)
        psf_tmp = np.zeros((szc, szc))
        if szc == 5:
            psf_tmp[2, 2] = 1
            psf_tmp[1, 2] = 1
            psf_tmp[2, 3] = 1
            psf_tmp[2, 1] = 1
            psf_tmp[3, 2] = 1
        elif szc == 7:
            psf_tmp[3, 3] = 1
            psf_tmp[2, 3] = 1
            psf_tmp[3, 2] = 1
            psf_tmp[4, 3] = 1
            psf_tmp[3, 4] = 1
        else:
            raise ValueError("szc not yet implemented")
        ilo = np.where(psf_tmp == 0)
        ihi = np.where(psf_tmp == 1)
    else:
        raise ValueError("Requested bp_shape not yet implemented.")

    frame_det = np.zeros_like(frame_corr)
    _, median, stddev = sigma_clipped_stats(hpf_frame, sigma=5,
                                            maxiters=None)
    bkg_full = median + 3*stddev
    for j in range(mszc, ny-mszc):
        for i in range(mszc, nx-mszc):
            ori = frame_corr[j-mszc:j+mszc+1, i-mszc:i+mszc+1]
            tmp = hpf_frame[j-mszc:j+mszc+1, i-mszc:i+mszc+1]
            # cross pixel intensities should all have same sign in filtered image
            if not np.all(tmp[ihi] > 0) and not np.all(tmp[ihi] < 0):
                frame_det[j, i] = 0
            # mean value in the cross should be larger than median+3*sigma over whole frame
            elif np.mean(tmp[ihi]) < bkg_full:
                frame_det[j, i] = 0
            # med/std in cross in non-filtered image should be at least 5
            elif np.median(ori[ihi])/np.std(ori) > 5:
                frame_det[j, i] = 0
            # bad pixel should match cross shape in both filtered and non-filtered image
            else:
                # score in filtered image
                num1 = np.nanmean(tmp[ihi])
                den1 = np.nanmean(tmp[ilo])
                if den1 == 0:
                    den1 = num1/10
                ratio = num1/den1
                num = np.sum(tmp[ihi])*ratio
                std_cross = np.std(tmp[ihi])
                if std_cross == 0:
                    std_cross = 1
                den = np.sum(psf_tmp)*std_cross
                score1 = num/den
                # score in original image
                num1 = np.nanmean(ori[ihi])
                den1 = np.nanmean(ori[ilo])
                if den1 == 0:
                    den1 = num1/10
                ratio = num1/den1
                num = np.sum(ori[ihi])*ratio
                std_cross = np.std(ori[ihi])
                if std_cross == 0:
                    std_cross = 1
                den = np.sum(psf_tmp)*std_cross
                score2 = num/den
                frame_det[j, i] = (score1+score2)/2
    # Estimation of background level
    _, median, stddev = sigma_clipped_stats(frame_det[np.where(frame_det > 0)],
                                            sigma=5,
                                            maxiters=None)
    bkg_level = median + (stddev * sig)
    if debug:
        print(
            'Sigma clipped median in TMP_detcorr_cross = {:.3f}'.format(median))
        print(
            'Sigma clipped stddev in TMP_detcorr_cross = {:.3f}'.format(stddev))
        print('Background threshold in TMP_detcorr_cross = {:.3f}'.format(
            bkg_level), '\n')
    # identification of bad-pixel crosses
    cross_det = np.zeros_like(frame_det)
    cross_det[np.where(frame_det > bkg_level)] = 1

    return frame_det, cross_det


def det_corr_line(frame_corr, hpf_frame, sig=5, sig_clip=5):

    ny, nx = frame_corr.shape

    # To minimize false positives, remove any (vertical) line detections, also by match-filtering
    szl = 5
    mszl = int((szl-1)/2)
    line_tmp = np.zeros((szl, szl))
    line_tmp[:, mszl] = 1
    ilo = np.where(line_tmp == 0)
    ihi = np.where(line_tmp == 1)

    frame_det_line = np.zeros_like(frame_corr)

    _, median, stddev = sigma_clipped_stats(hpf_frame, sigma=sig_clip,
                                            maxiters=None)
    bkg_full = median + 3*stddev

    for j in range(mszl, ny-mszl):
        for i in range(mszl, nx-mszl):
            tmp = hpf_frame[j-mszl:j+mszl+1, i-mszl:i+mszl+1]
            tmp_l = tmp[ihi]
            # mean value in the line should be larger than median+3*sigma over whole frame
            if np.mean(tmp[ihi]) < bkg_full:
                frame_det_line[j, i] = 0
            # mean value in the line should also be larger than mean in the rest of the box
            elif np.mean(tmp[ihi]) < np.mean(tmp[ilo]):
                frame_det_line[j, i] = 0
            # no more than 2 neg/zero values in the line
            elif np.sum(line_tmp[np.where(tmp_l == 0)]) > 2:
                frame_det_line[j, i] = 0
            # bad pixel should match line shape
            else:
                num1 = np.nanmean(tmp[ihi])
                den1 = np.nanmean(tmp[ilo])
                if den1 == 0:
                    den1 = num1/10
                ratio = num1/den1
                num = np.sum(tmp[ihi])*ratio
                std_cross = np.std(tmp[ihi])
                if std_cross == 0:
                    std_cross = 1
                den = np.sum(line_tmp)*std_cross
                frame_det_line[j, i] = num/den

    # Estimation of background level
    _, median, stddev = sigma_clipped_stats(frame_det_line[np.where(frame_det_line > 0)],
                                            sigma=5,
                                            maxiters=None)
    bkg_level = median + (stddev * sig)
    # identification of vertical lines
    lines_det = np.zeros_like(frame_det_line)
    lines_det[np.where(frame_det_line > bkg_level)] = 1

    return frame_det_line, lines_det


def bpix_corr2d(info_table, sig=5, max_nit=2, suffix='_bpc', bp_shape='+',
                kernel_sz=3, flag_only=True, verbose=True, debug=False,
                outpath=None, overwrite=True):
    """
    Corrects bad pixels in MIRI/MRS detector images. The routine uses an
    iterative sigma clipping algorithm for the identification of bad pixels,
    considering the DQ map as original bad pixel map estimate, and uses a
    Gaussian kernel for their correction.

    Parameters
    ----------
    info_table: table
        Table output from on the file list for which bad pixel correction
        should be performed.
    sig: float or int
        Sigma used for sigma-clipping
    protect_mask: float or int
        Radius of protection mask around star, in pixels
    max_nit: int, opt
        Maximum number of iterations for the iterative sigma clipping
    suffix: str, opt
        Suffix appended to original filename to save bad pixel corrected cube.
    bp_shape: str {'+'} or None, opt
        If not None, whether all new identified bad pixels should be expanded
        to a given shape of neighbouring pixels: '+' means that all pixels
        directly adjacent vertically or horizontally to identified bad pixels
        will also be considered bad and corrected.
    kernel_sz : int or None, optional
        Size of the high-pass filter Gaussian kernel used to better identify
        bad pixels.
    flag_only: bool, opt
        Whether to flag only the bad pixels, instead of correcting them.
    verbose: bool, opt
        Whether to print more information during processing.
    debug: bool, opt
        Whether to save an intermediate cube, used for the identification of
        bad pixels.
    outpath: str, opt
        If provided, path where output bad pixel corrected images are saved. By
        default, if not provided, will save files in the same folder as input.
    overwrite: bool, opt
        Whether to overwrite existing output files.

    Returns
    -------
    None (the output is saved as fits file)
    """

    for i in range(len(info_table)):
        fname = info_table['FILE_PATH'].iloc[i]
        frame_sci = fits.getdata(fname, 'SCI')
        channel_target, band_target, patt_num = info_table.iloc[i][['CHANNEL',
                                                                    'BAND',
                                                                    'PATT_NUM']]

        print("Processing {} ({}/{})".format(basename(fname), i+1, len(info_table)))

        if outpath is not None:
            outfname = outpath + \
                splitext(basename(fname))[0]+'{}.fits'.format(suffix)
        else:
            outpath = dirname(fname)
            outfname = splitext(fname)[0]+'{}.fits'.format(suffix)

        # First consider minimum of 4 dithers => identify static bad pixels in first BKG estimate
        bkg_tmp = "CH{}_B{}_bkg0.fits".format(channel_target, band_target)
        if not isfile(join(outpath, bkg_tmp)) or overwrite:
            # 1. Estimate common BKG from min intensities for each set of 4 dithers
            mask_same_CHB = np.logical_and.reduce([info_table['CHANNEL'] == channel_target,
                                                   info_table['BAND'] == band_target])
            dith_files = info_table[mask_same_CHB]['FILE_PATH'].values

            ndith = len(dith_files)
            if ndith != 2 and ndith != 4:
                raise ValueError("Only {} dither files found.".format(ndith))

            all_diths = np.empty([ndith, frame_sci.shape[0],
                                  frame_sci.shape[1]])
            for nd, df in enumerate(dith_files):
                all_diths[nd] = fits.getdata(df, 'SCI')

            bkg0 = np.amin(all_diths, axis=0)
            write_fits(join(outpath, bkg_tmp), bkg0)
        else:
            bkg0 = open_fits(bkg_tmp)

        if not isfile(outfname) or overwrite:

            # load fits file
            hdul = fits.open(fname)
            hdul.verify('ignore')
            frame = hdul[1].data
            dq = hdul[3].data

        #             ori_bp_map = dq.copy()
        #             ori_bp_map[np.where(ori_bp_map < dq_thr)] = 0
        #             ori_bp_map[np.where(ori_bp_map >= dq_thr)] = 1
        #             ori_bp_map[np.where(~np.isfinite(frame))] = 1

            DO_NOT_USE = dqflags.pixel['DO_NOT_USE']
            ori_bp_map = np.bitwise_and(dq, DO_NOT_USE) == DO_NOT_USE
            ori_bp_map[np.where(~np.isfinite(frame))] = True
            ori_bp_map[np.where(frame == 0.)] = True

            ori_bp_map = ori_bp_map.astype('int')
        #             if bp_val is not None:
        #                 if np.isscalar(bp_val):
        #                     bp_val = [bp_val]
        #                 for v in bp_val:
        #                     ori_bp_map[np.where(frame == v)] = 1

            # CHECK VIP version. If less than 1.3.2, update required to use "correct_only=False" option.
            if version.parse(vvip) < version.parse("1.3.3"):
                raise ValueError(
                    "Please update vip_hci package to a version larger or equal than 1.3.3")

            # identify additional bad pixels in bkg estimate
            frame_corr, bp_map1 = frame_fix_badpix_isolated(bkg0, sigma_clip=sig,
                                                            num_neig=5,
                                                            size=5,
                                                            verbose=False,
                                                            full_output=True)
            if debug:
                write_fits(outpath+"TMP_bp0.fits", ori_bp_map)
                write_fits(outpath+"TMP_bp1.fits", bp_map1)

            if bp_shape is not None:
                if bp_shape == '+':
                    bp_map1c = bp_map1.copy()
                    ny, nx = bp_map1.shape
                    for y in range(ny):
                        for x in range(nx):
                            if bp_map1[y, x]:
                                if y > 0:
                                    bp_map1c[y-1, x] = 1
                                if y < ny-1:
                                    bp_map1c[y+1, x] = 1
                                if x > 0:
                                    bp_map1c[y, x-1] = 1
                                if x < nx-1:
                                    bp_map1c[y, x+1] = 1
                    bp_map1 = bp_map1c.copy()
                else:
                    raise ValueError("Requested bp_shape not yet implemented.")

            if debug:
                write_fits(outpath+"TMP_bp1_cross.fits", bp_map1)

            # second pass to catch more bad pixels:
            bp_map2 = np.zeros_like(frame)
            if kernel_sz is not None and bp_shape is not None:
                # compute high-pass filter version of the image (remove smooth spectra)
                hpf_frame = frame_filter_highpass(frame_corr, mode='median-subt',
                                                  median_size=kernel_sz,
                                                  conv_mode='conv')
                hpf_frame[np.where(hpf_frame < 0)] = 0
                if debug:
                    write_fits(outpath+"TMP_hpf{}_frame.fits".format(kernel_sz),
                               hpf_frame)

                # identify crosses of 5 bad pixels by match filtering: these are additional bad pixels.
                frame_det, cross_det = det_corr_cross(frame_corr, hpf_frame,
                                                      bp_shape='+', sig=sig,
                                                      debug=debug)

                if debug:
                    write_fits(outpath+"TMP_corr_cross.fits", frame_det)
                    write_fits(outpath+"TMP_detcorr_cross.fits",
                               cross_det, verbose=False)

                # identify vertical lines of 5 pixels by cross-correlation: these are not bad pixels.
                frame_det_line, lines_det = det_corr_line(frame_corr, hpf_frame,
                                                          sig=sig)

                # consider final bad pixels as those not corresponding to line detections
                cross_det[np.where(lines_det)] = 0
                # add cross shape for bad pixel locations
                bp_map2 = cross_det.copy()
                ny, nx = cross_det.shape
                for y in range(ny):
                    for x in range(nx):
                        if cross_det[y, x]:
                            if y > 0:
                                bp_map2[y-1, x] = 1
                            if y < ny-1:
                                bp_map2[y+1, x] = 1
                            if x > 0:
                                bp_map2[y, x-1] = 1
                            if x < nx-1:
                                bp_map2[y, x+1] = 1

                if debug:
                    write_fits(outpath+"TMP_corr_line.fits",
                               frame_det_line, verbose=False)
                    write_fits(outpath+"TMP_detcorr_line.fits",
                               lines_det, verbose=False)
                    write_fits(outpath+"TMP_bp2.fits", bp_map2, verbose=False)

            bp_map_tmp = ori_bp_map + bp_map1 + bp_map2
            bp_map_tmp[np.where(bp_map_tmp > 1)] = 1
            frame_corr, bpix_map = cube_fix_badpix_clump(frame, bpm_mask=bp_map_tmp,
                                                         correct_only=True,
                                                         fwhm=6, sig=sig,
                                                         protect_mask=None,
                                                         verbose=verbose,
                                                         max_nit=max_nit, mad=False,
                                                         full_output=True)

            # last pass to catch even more bad pixels:
            # identify additional isolated bad pixels (not cross-shaped):
            frame_corr, bp_map3 = frame_fix_badpix_isolated(frame_corr, sigma_clip=sig,
                                                            num_neig=5,
                                                            size=5,
                                                            verbose=False,
                                                            full_output=True)
            if debug:
                write_fits(outpath+"TMP_bp3.fits", bp_map3)

            # final correction of bad pixels with Gaussian kernel
            fbp_map = ori_bp_map + bp_map1 + bp_map2 + bp_map3
            fbp_map[np.where(fbp_map > 1)] = 1

            if debug:
                write_fits(outpath+"TMP_fbp0123.fits", fbp_map, verbose=False)

            if flag_only:
                #                 ndq = dq.copy()
                #                 ndq[np.where(fbp_map)] += int(dq_thr+1)
                #                 hdul[3].data = ndq
                fbp_map_mask = fbp_map.astype('bool')
                dq_image = fbp_map_mask.astype(np.uint32)
                ndq = np.bitwise_or(dq, dq_image)
                hdul[3].data = ndq

            else:
                # final correction (in original frame) of all identified bad pixels, using Gaussian kernel
                frame_corr = cube_fix_badpix_interp(frame, fbp_map, mode='gauss',
                                                    fwhm=(4, 1))

                # update HDU and write fits
                hdul[1].data = frame_corr.copy()

            hdul.writeto(outfname, overwrite=True)
            hdul.close()


def bpc2(filename, outdir, endswith="rate_bpc_cal.fits", fwhm_sz=(4, 1), sig=5,
         nit_bp=3, flag_only=False, overwrite=True, debug=False):

    bname = basename(filename)
    ndig = len(endswith)
    # remove "rate_bpc_cal.fits"
    outname = join(outdir, bname[:-ndig]+"bpc2.fits")

    if not isfile(outname) or overwrite:

        hdul = fits.open(filename)
        hdul.verify('ignore')
        frame = hdul[1].data
        dq = hdul[3].data

        # discard non-finite values
        bpmap_ori = np.zeros_like(frame).astype('bool')
        bpmap_ori[np.where(~np.isfinite(frame))] = True
        bpmap_ori[np.where(frame == 0.)] = True

        # set pixels with non-finite values or zero as DO_NOT_USE (in case this is not already the case)
#         dq_image = bpmap_ori.astype(np.uint32)
#         ndq =  np.bitwise_or(hdul[3].data, dq_image)
#         hdul[3].data = ndq

        DO_NOT_USE = dqflags.pixel['DO_NOT_USE']
        NON_SCIENCE = dqflags.pixel['NON_SCIENCE']

        mask_bad = np.bitwise_and(dq, DO_NOT_USE) == DO_NOT_USE
        mask_non_science = np.bitwise_and(dq, NON_SCIENCE) == NON_SCIENCE

        bpmap_ori = (mask_bad & ~mask_non_science) | (
            bpmap_ori & mask_non_science)

        bpmap_ori = bpmap_ori.astype('int')

        if debug:
            fits.writeto(
                join(outdir, bname[:-ndig] +
                     "init_do_not_use.fits"), mask_bad.astype('int'),
                overwrite=True)
            fits.writeto(
                join(
                    outdir, bname[:-ndig]+"init_not_science.fits"), mask_non_science.astype('int'),
                overwrite=True)

        frame_nan = frame.copy()
        frame_nan[mask_non_science] = np.nan
        # interp NaNs
        frame_interp = interp_nan(frame_nan, Gaussian2DKernel(x_stddev=fwhm_sz[0],
                                                              y_stddev=fwhm_sz[1],
                                                              x_size=None,
                                                              y_size=None),
                                  convolve=convolve)

        if np.sum(np.isnan(frame_interp)) > 0:
            frame_interp = interp_nan(frame_interp,
                                      Gaussian2DKernel(x_stddev=fwhm_sz[0],
                                                       y_stddev=fwhm_sz[1],
                                                       x_size=None,
                                                       y_size=None),
                                      convolve=convolve)

        if debug:
            fits.writeto(
                join(outdir, bname[:-ndig]+"frame_interp.fits"), frame_interp)

        _, bpix_map = cube_fix_badpix_clump(frame_interp, bpm_mask=bpmap_ori,
                                            fwhm=4, sig=sig, verbose=True,
                                            max_nit=nit_bp, mad=False,
                                            full_output=True)

        _, bpix_map = cube_fix_badpix_clump(frame_interp, bpm_mask=bpix_map,
                                            fwhm=6, sig=sig, verbose=True,
                                            max_nit=2, mad=False,
                                            full_output=True)

        frame_corr = cube_fix_badpix_interp(frame_interp, bpix_map, mode='gauss',
                                            fwhm=fwhm_sz)

        if flag_only:
            dq_image = bpix_map.astype(np.uint32)
            ndq = np.bitwise_or(dq, dq_image)
            hdul[3].data = ndq
        else:
            frame_fin = frame.copy()
            frame_fin[np.where(bpix_map)] = frame_corr[np.where(bpix_map)]
            hdul[1].data = frame_fin

        if debug:
            #             write_fits(outdir+"TMP_{}_bpmap2.fits".format(bname), bpix_map)
            fits.writeto(
                join(outdir, bname[:-ndig]+"bpmap_final.fits"), bpix_map.astype('int'), overwrite=True)

        hdul.writeto(outname, overwrite=True)
        hdul.close()


def bpix_corr3d(fname, sig=5, protect_mask=5, max_nit=5, suffix='_bpc',
                verbose=True, debug=False, overwrite=True):
    """
    Corrects bad pixels in MIRI/MRS s3d cubes from either stage 2 or stage 3, 
    using an iterative sigma clipping algorithm and protecting the star for 
    the identification, and using a Gaussian kernel for the correction.

    Parameters
    ----------
    fname: str
        filename
    sig: float or int
        Sigma used for sigma-clipping
    protect_mask: float or int
        Radius of protection mask around star, in pixels
    max_nit: int, opt
        Maximum number of iterations for the iterative sigma clipping
    suffix: str, opt
        Suffix appended to original filename to save bad pixel corrected cube.
    verbose: bool, opt
        Whether to print more information during processing.
    debug: bool, opt
        Whether to save an intermediate cube, used for the identification of 
        bad pixels.

    Returns
    -------
    None (the output is saved as fits file)
    """

    outfname = fname[:-5]+'{}.fits'.format(suffix)

    if not isfile(outfname) or overwrite:
        if verbose:
            print("*** Correcting bad pixels of {} ***".format(basename(fname)))
        # load fits file
        hdul = fits.open(fname)
        hdul.verify('ignore')
        cube = hdul[1].data
        dq = hdul[3].data

        # Identify non-zero channels (can be present in spec2 s3d cubes]
        good_idx = np.where(np.nanmean(
            dq.reshape(dq.shape[0], -1), axis=1) != 513)[0]
        # if verbose:
        #    print("In total there are {:.0f} good channels for bad pixel correction".format(len(good_idx)))
        good_cube = cube[good_idx].copy()
        cen_cube = cube[good_idx].copy()
        good_dq = dq[good_idx].copy()
        cube_corr = cube.copy()
        dq_corr = np.zeros_like(cube_corr)
        ori_mask = np.zeros_like(cube_corr)
        ori_mask[np.where(dq_corr == 0)] = 1

        # interpolate zero-padding with Gaussian kernel to have smooth edges
        # here only for centering => all neg values interpolated
        cen_cube[np.where(good_cube <= 0)] = np.nan
        for z, ch in enumerate(good_idx):
            cube_corr[ch] = interp_nan(cen_cube[z], Gaussian2DKernel(x_stddev=1.,
                                                                     y_stddev=1.,
                                                                     x_size=None,
                                                                     y_size=None),
                                       convolve=convolve)
            cen_cube[z] = cube_corr[ch].copy()
        cen_cube[np.where(np.isnan(cen_cube))] = 0
        cen_cube = mask_circle(cen_cube, 10, 0, mode='out')

        # find approximate star location in each cube
        star_approx_idx = approx_stellar_position(
            cen_cube, fwhm=4, return_test=False, verbose=debug)
        cy = star_approx_idx[:, 0]
        cx = star_approx_idx[:, 1]

        # interpolate zero-padding with Gaussian kernel to have smooth edges
        good_cube[np.where(good_cube == 0)] = np.nan
        for z, ch in enumerate(good_idx):
            cube_corr[ch] = interp_nan(good_cube[z], Gaussian2DKernel(x_stddev=1.,
                                                                      y_stddev=1.,
                                                                      x_size=None,
                                                                      y_size=None),
                                       convolve=convolve)
            good_cube[z] = cube_corr[ch].copy()
        good_cube[np.where(np.isnan(good_cube))] = 0

        # identify bad pixels with iterative sigma clipping and a protect mask to avoid star
        cube_corr, bpix_map = cube_fix_badpix_clump(good_cube, bpm_mask=None,
                                                    cy=cy, cx=cx,
                                                    fwhm=3, sig=sig,
                                                    protect_mask=protect_mask,
                                                    verbose=debug,
                                                    # min_thr=min_thr,
                                                    max_nit=max_nit, mad=False,
                                                    full_output=True)

        if debug:
            hdul[1].data = good_cube.copy()
            hdul.writeto(splitext(fname)[0]+'_smooth.fits', overwrite=True)
            hdul[1].data = bpix_map.copy()
            hdul.writeto(splitext(fname)[0]+'_bpn.fits', overwrite=True)

        # correct bad pixels with Gaussian kernel
        good_cube = cube_fix_badpix_interp(good_cube, bpix_map, mode='gauss',
                                           fwhm=3)
        if debug:
            hdul[1].data = good_cube.copy()
            hdul.writeto(splitext(fname)[0]+'_postsmooth.fits', overwrite=True)

        if verbose:
            msg = "A total of {} bad pixels were found and corrected."
            print(msg.format(int(np.sum(bpix_map))))

        # write output cube
        for z, ch in enumerate(good_idx):
            cube_corr[ch] = good_cube[z]*ori_mask[z]
            dq_corr[ch] = good_dq[z]

        # update HDU and write fits
        hdul[1].data = cube_corr.copy()
        hdul[3].data = dq_corr.copy()
        hdul.writeto(fname[:-5]+'{}.fits'.format(suffix), overwrite=True)
        hdul.close()
