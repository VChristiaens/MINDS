#! /usr/bin/env python
"""
Module with helper functions for pair-wise dither subtraction or dedicated BKG
subtraction on detector images.

More information here: https://vip.readthedocs.io/
"""
__author__ = "M. Temmink"
__all__ = [
    "Baseline",
    "PureBaseline",
    "PlotSpectrum",
]

import matplotlib.pyplot as plt
import numpy as np
import pybaselines as pb
from astropy.stats import sigma_clipped_stats as scs
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d as I1D
from scipy.signal import savgol_filter as sf
from spectres import spectres


def Baseline(Wavelength = [],
             Flux       = [],
             WL         = 100,
             NK         = 100,
             Quant      = 0.05,
             SD         = 3,
             DO         = 3,
             MI         = 100,
             Lam        = 1e2,
             Tol        = 1e-6,
             sigma_clip = (2, 3),
             W_max_out = 27.8):
    ## Determining the baseline of JWST spectra using a consistent way:
    ## https://pybaselines.readthedocs.io/en/latest/

    ## Outlier detection:
    W, F = Wavelength, Flux
    W_clip = Wavelength[np.where(Wavelength < W_max_out)]
    F_clip = Flux[np.where(Wavelength < W_max_out)]

    Outliers = 1
    while Outliers > 0 & WL > len(F):
        Filtered = sf(F, window_length=WL, polyorder=3)
        Filtered_clip = sf(F_clip, window_length=WL, polyorder=3)
        STD = scs(F_clip - Filtered_clip, sigma=3)[2]
        Mask = (F - Filtered) > sigma_clip[0] * STD  ## 2 Sigma lines are masked

        Outliers = len(F[Mask])
        print(Outliers)
        if Outliers > 0:
            W, F = W[~Mask], F[~Mask]
            pass
        else:
            pass
        pass

    W_clip = W[np.where(W < W_max_out)]
    F_clip = F[np.where(W < W_max_out)]

    Filtered = sf(F, window_length=WL, polyorder=3)
    Filtered_clip = sf(F_clip, window_length=WL, polyorder=3)
    STD = scs(F_clip-Filtered_clip, sigma=3)[2]  ## STD over the masked Savgol-filter.
    FilterFull = spectres(Wavelength, W, Filtered)
    Mask = (Flux-FilterFull) < -sigma_clip[1]*STD  ## Masking all 3sigma downwards spikes.

    Baseline = pb.spline.irsqr(x_data        = Wavelength[~Mask],
                               data          = Flux[~Mask],
                               num_knots     = NK,
                               quantile      = Quant,
                               spline_degree = SD,
                               diff_order    = DO,
                               max_iter      = MI,
                               lam           = Lam,
                               tol           = Tol)[0]
    Baseline = spectres(Wavelength, Wavelength[~Mask], Baseline)
    return Baseline, Mask

def PureBaseline(Wavelength = [],
                 MaskedWav  = [],
                 MaskedFlux = [],
                 NK         = 100,
                 Quant      = 0.05,
                 SD         = 3,
                 DO         = 3,
                 MI         = 100,
                 Lam        = 1e2,
                 Tol        = 1e-6):
    Baseline = pb.spline.irsqr(x_data        = MaskedWav,
                               data          = MaskedFlux,
                               num_knots     = NK,
                               quantile      = Quant,
                               spline_degree = SD,
                               diff_order    = DO,
                               max_iter      = MI,
                               lam           = Lam,
                               tol           = Tol)[0]
    Baseline = spectres(Wavelength, MaskedWav, Baseline)
    return Baseline

def PlotSpectrum(Wav      = [],
                 Flux     = [],
                 Baseline = [],
                 Mask     = [],
                 dY       = 0.25,
                 SavePlot = '',
                 MaskFeature=True):

    XL = [np.nanmin(Wav), np.nanmax(Wav)]
    YL = [np.nanmin(Flux), np.nanmax(Flux)+dY]

    if len(Baseline) > 0:
        fig, ax = plt.subplots(2, 1, figsize=(10,7))

        ax[0].step(Wav, Flux, color='k', where='mid', zorder=5)
        ax[0].plot(Wav, Baseline, color='firebrick', zorder=5)
        ax[0].scatter(Wav[Mask], Flux[Mask], color='red', marker='x', s=50)

        ax[1].step(Wav, Flux-Baseline, color='k', where='mid', zorder=5)
        ax[1].axhline(y=0, color='firebrick', ls='dashed')

        ax[0].set(xlim=XL, ylim=YL, xticklabels=[])
        YL = [np.nanmin(Flux-Baseline), np.nanmax(Flux-Baseline)+dY]
        ax[1].set(xlabel=r'Wavelength [$\mu$m]', ylabel='Flux [Jy]', xlim=XL,
                  ylim=YL)
        pass
    else:
        fig, ax = plt.subplots(figsize=(10,4))

        ax.step(Wav, Flux, color='k', where='mid')
        ax.set(xlabel=r'Wavelength [$\mu$m]', ylabel='Flux [Jy]', xlim=XL,
               ylim=YL)
        pass

    for i in range(len(ax)):
        ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].yaxis.set_minor_locator(AutoMinorLocator(5))
        pass

    if MaskFeature:
        FMask         = (Wav >= 12.118) & (Wav <= 12.132)
        Interpolation = I1D(x=Wav[~FMask], y=Flux[~FMask], kind='cubic')
        FeatureFlux   = Interpolation(Wav)
        ax[0].step(Wav, FeatureFlux, color='darkslateblue', where='mid',
                   alpha=0.6)
        ax[1].step(Wav, FeatureFlux-Baseline, color='darkslateblue',
                   where='mid', alpha=0.6)
        pass

    ## --- Plot the blue and red ends over the overlapping wavelength regions:
    Reds  = [5.74, 6.63, 7.65, 8.77, 10.13, 11.70, 13.47, 15.57, 17.98, 20.95,
             24.48]
    Blues = [5.66, 6.53, 7.51, 8.67, 10.02, 11.55, 13.34, 15.41, 17.70, 20.69,
             24.19]

    for i in range(len(Reds)):
        ax[0].axvline(x=Reds[i], color='red', lw=2, alpha=0.6, ls='dashed')
        ax[1].axvline(x=Reds[i], color='red', lw=2, alpha=0.6, ls='dashed')
        ax[0].axvline(x=Blues[i], color='royalblue', lw=2, alpha=0.6, ls='dashed')
        ax[1].axvline(x=Blues[i], color='royalblue', lw=2, alpha=0.6, ls='dashed')
        pass

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(f'{SavePlot}.png', dpi=250)
    plt.show()
    #plt.close()
    #pass