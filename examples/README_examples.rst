pre_v1.14:
----------

Example MINDS notebooks for jwst pipeline versions up to v1.14 (by default the do_bpc2 flag is on and the outlier detection step is skipped):

- MINDS_reduction_a.ipynb : Example of reduction with default parameters. The background estimation and subtraction is only done at the moment of spectrum extraction, in the frames of the final spectral cube, The background estimate is made in an annulus around the source, and the subtraction compensates for the amount of PSF flux expected to be found in the annulus used for the background estimation. By default the results are saved in folders with suffix "_MINDSv1.0.3a".
- MINDS_reductions_sdither.ipynb : Example of reduction with the "smooth dither" background estimation. For a 4-point dither observation, this method estimates the background in detector images, as the minimum of the 4 dither images, further smoothed with a median filter. The latter aims to further remove any remaining PSF signal (which could appear as thin lines on the detector) from the background estimate, while also performing a low-pass filter.
- MINDS_reduction_multi.ipynb : Example reduction for multiple objects - otherwise similar to MINDS_reduction.ipynb


v1.14:
------
Example MINDS notebooks for jwst pipeline versions for v1.14 and beyond (by default the do_bpc2 flag is off and the outlier detection step is on):

- MINDS_reduction_b.ipynb : Example of reduction with default parameters. The background estimation and subtraction is only done at the moment of spectrum extraction, in the frames of the final spectral cube, The background estimate is made in an annulus around the source, and the subtraction compensates for the amount of PSF flux expected to be found in the annulus used for the background estimation. By default the results are saved in folders with suffix "_MINDSv1.0.3b".


jwst_query.cfg:
---------------
Example config file to download uncal files from MAST for your target, which are then used as input to the MINDS pipeline.