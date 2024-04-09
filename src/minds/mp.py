#! /usr/bin/env python
"""
Module with helper functions for pair-wise dither subtraction or dedicated BKG
subtraction on detector images.

More information here: https://vip.readthedocs.io/
"""
__author__ = "M. Samland, V. Christiaens"
__all__ = [
    "runmany",
    "add_dq",
    "rundet1",
    "run_straylight",
    "runspec2"
]

from functools import partial

from astropy.io import fits

try:
    from ray.util.multiprocessing import Pool
except:
    from multiprocessing import Pool
import os
from os.path import basename, join, splitext

import jwst
from jwst.pipeline import Detector1Pipeline, Spec2Pipeline
from packaging import version

vjwst = jwst.__version__
# print("JWST pipeline version: ", vjwst)

if version.parse(vjwst) < version.parse("1.13.0"):
    raise ValueError(
        "Please update JWST package to a version larger or equal than 1.13.0"
    )

def runmany(maxp, step, filenames, **extra_args):
    #if __name__ == '__main__':
    p = Pool(maxp)
    # res = p.map(step, filenames)
    p.map(partial(step, **extra_args), filenames)
    p.close()
    p.join()

def add_dq(filename, outdir):
    print(filename)

    # skip if file exists and no overwrite is desired:
    print(os.path.join(outdir, filename.replace('uncal.fits', 'dq.fits')))

    det1 = Detector1Pipeline()  # Instantiate the pipeline

    det1.output_dir = outdir  # Specify where the output should go

    # New set of recommended parameters (cfr. data reduction meeting Oct. 27)

    # Overrides for whether or not certain steps should be skipped
    det1.dq_init.skip = False
    det1.dq_init.suffix = "dq"
    det1.dq_init.save_results = True
    det1.saturation.skip = True
    det1.firstframe.skip = True
    det1.lastframe.skip = True
    det1.reset.skip = True
    det1.linearity.skip = True
    det1.rscd.skip = True

    det1.dark_current.skip = True
    det1.refpix.skip = True
    det1.jump.skip = True
    det1.ramp_fit.skip = True
    det1.gain_scale.skip = True

    det1.save_results = False  # Save the final resulting _rate.fits files
    det1(filename)  # Run the pipeline on an input list of files


def rundet1(filename, outdir, use_agg_det1_params=False):
    print(filename)

    ## skip if file exists and no overwrite is desired:
    print(join(outdir, basename(splitext(filename)[0][:-5]), 'rate.fits'))

    # This initial setup is just to make sure that we get the latest parameter reference files
    # pulled in for our files.  This is a temporary workaround to get around an issue with
    # how this pipeline calling method works.
    crds_config = Detector1Pipeline.get_config_from_reference(filename)
    det1 = Detector1Pipeline.from_config_section(crds_config)
    #det1 = Detector1Pipeline() # Instantiate the pipeline

    det1.output_dir = outdir # Specify where the output should go

    # New set of recommended parameters (cfr. data reduction meeting Oct. 27)
    if use_agg_det1_params:
        det1.jump.rejection_threshold = 4
        det1.jump.expand_large_events = True
        det1.jump.min_jump_area = 15
        det1.jump.use_ellipses = True
        det1.jump.expand_factor = 3
        det1.jump.after_jump_flag_dn1 = 10
        det1.jump.after_jump_flag_time1 = 20
        det1.jump.after_jump_flag_dn2 = 1000
        det1.jump.after_jump_flag_time2 = 3000

    # Overrides for whether or not certain steps should be skipped
    # det1.dq_init.skip = False
    # det1.saturation.skip = False
    # det1.firstframe.skip = False
    # det1.lastframe.skip = False
    # det1.reset.skip = False
    # det1.linearity.skip = False
    # det1.rscd.skip = False
    # det1.dark_current.skip = False
    # det1.refpix.skip = False
    # det1.jump.skip = False
    # det1.ramp_fit.skip = False
    # det1.gain_scale.skip = False

    # Bad pixel mask overrides
    # det1.dq_init.override_mask = 'myfile.fits'

    # Saturation overrides
    # det1.saturation.override_saturation = 'myfile.fits'

    # Reset overrides
    # det1.reset.override_reset = 'myfile.fits'

    # Linearity overrides
    # det1.linearity.override_linearity = 'myfile.fits'

    # RSCD overrides
    # det1.rscd.override_rscd = 'myfile.fits'

    # DARK overrides
    # det1.dark_current.override_dark = 'myfile.fits'

    # GAIN overrides
    # det1.jump.override_gain = 'myfile.fits'
    # det1.ramp_fit.override_gain = 'myfile.fits'

    # READNOISE overrides
    # det1.jump.override_readnoise = 'myfile.fits'
    # det1.ramp_fit.override_readnoise = 'myfile.fits'

    det1.save_results = True # Save the final resulting _rate.fits files
    det1(filename) # Run the pipeline on an input list of files


def run_straylight(filename, outdir):
    crds_config = Spec2Pipeline().get_config_from_reference(filename)
    spec2 = Spec2Pipeline().from_config_section(crds_config)
#     spec2 = Spec2Pipeline()
    spec2.output_dir = outdir

    # Straylight overrides
    # spec2.straylight.override_mrsxartcorr = 'myfile.fits'

    # Overrides for whether or not certain steps should be skipped
    spec2.assign_wcs.skip = False
    spec2.bkg_subtract.skip = True
    spec2.flat_field.skip = True
    spec2.srctype.skip = True
    spec2.straylight.skip = False
    spec2.straylight.suffix = "straylight"
    spec2.straylight.save_results = True
    spec2.fringe.skip = True
    spec2.photom.skip = True
    spec2.cube_build.skip = True
    spec2.extract_1d.skip = True

    # Do we need to set this false to prevent _cal file to be created?
    # spec2.save_results = True
    spec2(filename)


def runspec2(filename, outdir, psff=False, psff_dir='./psff_ref/',
             phot_ver='9B.04.19', fringe_ver='2', pixel_replace_algo='mingrad',
             dith_combi_method='drizzle'):
    # This initial setup is just to make sure that we get the latest parameter reference files
    # pulled in for our files.  This is a temporary workaround to get around an issue with
    # how this pipeline calling method works.
    crds_config = Spec2Pipeline.get_config_from_reference(filename)
    spec2 = Spec2Pipeline.from_config_section(crds_config)
    #spec2 = Spec2Pipeline() # Instantiate the pipeline

    spec2.output_dir = outdir

    # PSFF overrides
    if psff:
        hdu = fits.open(filename)
        channel = hdu[0].header['CHANNEL']
        band = hdu[0].header['BAND']
        ifu = hdu[0].header['DETECTOR']
        dith_num = hdu[0].header['PATT_NUM']
        dith_dir = hdu[0].header['DITHDIRC']
        hdu.close()

        if band=='SHORT':
            subband='A'
        elif band=='MEDIUM':
            subband='B'
        elif band=='LONG':
            subband='C'

        if dith_dir=='NEGATIVE':
            dith = 'neg'
        elif dith_dir=='POSITIVE':
            dith = 'pos'

        spec2.photom.override_photom = join(psff_dir,
                                            'PHOTOM/MIRI_FM_{}_{}_PHOTOM_{}_{}dither.fits'.format(ifu, channel+band, phot_ver, dith))
        spec2.fringe.override_fringe = join(psff_dir,
                                            'FRINGE/point_source_fringe_flat_{}dither{}_{}_v{}.fits'.format(dith, dith_num, channel+subband, fringe_ver))
        spec2.flat_field.skip = True

    # Assign_wcs overrides
    # spec2.assign_wcs.override_distortion = 'myfile.asdf'
    # spec2.assign_wcs.override_regions = 'myfile.asdf'
    # spec2.assign_wcs.override_specwcs = 'myfile.asdf'
    # spec2.assign_wcs.override_wavelengthrange = 'myfile.asdf'

    # Flatfield overrides
    # spec2.flat.override_flat = 'myfile.fits'
    else:
        spec2.flat_field.skip = False

    # Background
    spec2.bkg_subtract.skip = True

    # Straylight overrides
    # spec2.straylight.override_mrsxartcorr = 'myfile.fits'
    spec2.straylight.skip = True  # starting from v6e!

    # Fringe overrides
    spec2.fringe.skip = False

    # Photom overrides
    # spec2.photom.override_photom = 'myfile.fits'
    spec2.photom.skip = False

    # Residual fringe correction
    if psff:
        spec2.residual_fringe.skip = True
    else:
        spec2.residual_fringe.skip = False

    # Bad pixel correction
    spec2.pixel_replace.skip = False
    spec2.pixel_replace.algorithm = pixel_replace_algo

    # Cubepar overrides
    spec2.cube_build.skip = True
    # spec2.cube_build.override_cubepar = 'myfile.fits'

    # Extract1D overrides
    # spec2.extract1d.skip = True
    # spec2.extract1d.override_extract1d = 'myfile.asdf'
    # spec2.extract1d.override_apcorr = 'myfile.asdf'

    # Some cube building options
    spec2.cube_build.weighting = dith_combi_method
    spec2.cube_build.coord_system = 'skyalign'
    spec2.cube_build.output_type = 'channel'

    spec2.save_results = True
    spec2(filename)
