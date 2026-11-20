#!/usr/bin/env python3
"""Render the Na I D region or full VIS coverage of KOBE-1 (KOBE-056)."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d, percentile_filter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrum", type=Path, help="KOBE-056 merged FITS file")
    parser.add_argument("output", type=Path, help="Output SVG path")
    parser.add_argument("--mobile", action="store_true", help="Use a taller layout and larger axis labels")
    parser.add_argument("--full-spectrum", action="store_true", help="Show full VIS coverage with synthesis windows")
    parser.add_argument("--ccf-mask", type=Path, help="Overlay the real K5 mask on the 616–624 nm spectrum")
    args = parser.parse_args()

    with fits.open(args.spectrum) as hdus:
        wavelength = np.asarray(hdus["WAVELENGTH"].data, dtype=float) / 10  # Å to nm
        flux = np.asarray(hdus["FLUX"].data, dtype=float)

    valid = np.isfinite(wavelength) & np.isfinite(flux) & (flux > 0)
    wavelength, flux = wavelength[valid], flux[valid]
    lower, upper = (616.0, 624.0) if args.ccf_mask else (584.0, 594.0)
    region = np.ones(wavelength.size, dtype=bool) if args.full_spectrum else (wavelength >= lower - 2) & (wavelength <= upper + 2)
    wavelength, flux = wavelength[region], flux[region]
    if wavelength.size < 100:
        raise ValueError("The sodium doublet is outside this spectrum")
    continuum = gaussian_filter1d(percentile_filter(flux, 95, size=801), 60)
    relative_flux = flux / continuum
    window = np.ones(wavelength.size, dtype=bool) if args.full_spectrum else (wavelength >= lower) & (wavelength <= upper)

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
    fig, ax = plt.subplots(figsize=(5.4, 4.1) if args.mobile else (11, 4.4), dpi=160)
    fig.patch.set_facecolor("#101115")
    ax.set_facecolor("#101115")
    if args.ccf_mask:
        mask = np.genfromtxt(args.ccf_mask, names=True, delimiter="\t")
        mask = mask[(mask['wave_peak'] >= lower) & (mask['wave_peak'] <= upper) & (mask['depth'] >= 0.01)]
        if not mask.size:
            raise ValueError("No K5 mask lines in this interval")
        ax.plot(wavelength[window], relative_flux[window], color="#dfe5e5", lw=1.05)
        ax.vlines(mask['wave_peak'], 0, -0.24 * mask['depth'] / mask['depth'].max(),
                  color="#8bc6a5", lw=1.3)
        ax.axhline(0, color="#32383d", lw=0.6)
        ax.set_xlim(lower, upper)
    elif args.full_spectrum:
        for lower, upper in [(562, 686), (695, 713), (737, 756), (773, 807), (837, 860)]:
            ax.axvspan(lower, upper, color="#8bc6a5", alpha=0.20, lw=0)
        # Preserve both extrema in each display bin so narrow absorption lines
        # survive the overview without shipping every FITS pixel in the SVG.
        bins = np.array_split(np.arange(wavelength.size), 1600 if not args.mobile else 800)
        indices = np.sort(np.concatenate([
            [chunk[np.argmin(relative_flux[chunk])], chunk[np.argmax(relative_flux[chunk])]]
            for chunk in bins if chunk.size
        ]))
        ax.plot(wavelength[indices], relative_flux[indices], color="#dfe5e5", lw=0.55)
        ax.set_xlim(561.8, 919.4)
    else:
        for center in (588.995, 589.592):
            ax.axvline(center, ymin=0, ymax=0.91, color="#8bc6a5",
                       lw=1.1, linestyle=(0, (4, 4)), alpha=0.8)
        ax.plot(wavelength[window], relative_flux[window], color="#dfe5e5", lw=1.05)
        ax.set_xlim(584.0, 594.0)
    ax.set_ylim(-0.28 if args.ccf_mask else 0, 1.14)
    if not args.full_spectrum and not args.ccf_mask:
        ax.text(589.3, 1.08, "Na I D", ha="center", va="center", color="#a6cbb7",
                fontsize=19 if args.mobile else 15)
    label_size = 21 if args.mobile else 16
    ax.set_xlabel("Wavelength · nm", color="#b9c0c2", labelpad=16, fontsize=label_size)
    ax.set_ylabel("Flux" if args.mobile else "Relative flux", color="#b9c0c2",
                  labelpad=16, fontsize=label_size)
    ax.tick_params(colors="#aeb6b9", labelsize=18 if args.mobile else 14, length=0, pad=11)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([616, 618, 620, 622, 624] if args.ccf_mask else
                 [600, 700, 800, 900] if args.full_spectrum else [584, 586, 588, 590, 592, 594])
    ax.grid(axis="y", color="#32383d", lw=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.20 if args.mobile else 0.10, right=0.94 if args.mobile else 0.98,
                        top=0.91, bottom=0.23 if args.mobile else 0.21)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=fig.get_facecolor())
    plt.close(fig)
    args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
