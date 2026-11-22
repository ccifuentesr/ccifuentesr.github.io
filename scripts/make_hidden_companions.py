"""Recompute the portfolio example from local FIES or HERMES spectra using project_one.

Run with the iSpec environment. Source FITS files stay outside the website.
CCF extraction follows the existing project pipeline; VHELIO defines a
heliocentric (not barycentric) velocity frame. S/N >= 10 is its draft cut.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from dataclasses import asdict, replace

parser = argparse.ArgumentParser()
parser.add_argument('--pipeline', type=Path, required=True)
parser.add_argument('--spectra', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--instrument', choices=['FIES','HERMES'], default='FIES')
parser.add_argument('--min-snr', type=float, default=10)
parser.add_argument('--target', default='RX J1417.5+0233')
parser.add_argument('--pattern', default='RX_J1417_*.fits')
args = parser.parse_args()
spec = importlib.util.spec_from_file_location('one_ccf', args.pipeline)
pipeline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pipeline
spec.loader.exec_module(pipeline)
import pandas as pd
import numpy as np
# These observations used F3 MedRes; use its resolving power for continuum fitting.
configs = pipeline.instrument_configs()
configs['FIES'] = replace(configs['FIES'], resolution=46000)
config = configs[args.instrument]
pipeline.instrument_configs = lambda: configs
rows = []
for path in sorted(args.spectra.glob(args.pattern)):
    metadata = pipeline.read_metadata(path, config)
    header_correction = metadata.barycorr_kms
    if args.instrument == 'HERMES':
        # DATA/data_one/reference.txt documents these log_merged_c_lin inputs
        # as already barycentric. Apply only the barycentric-to-heliocentric
        # difference, not the full observer correction a second time.
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        import astropy.units as u
        time = Time(metadata.date_obs, scale='utc') + (metadata.exptime_s or 0)/2*u.s
        coord = SkyCoord(ra=metadata.ra_deg*u.deg, dec=metadata.dec_deg*u.deg)
        correction = coord.radial_velocity_correction(kind='heliocentric', obstime=time, location=config.site).to_value(u.km/u.s)
        metadata = replace(metadata, barycorr_kms=float(correction-header_correction), barycentric_applied=False)
    row = pd.Series(asdict(metadata))
    spectrum, info = pipeline.prepare_normalized_spectrum(row)
    item = {'instrument':args.instrument, 'file': path.name, 'date': metadata.date_obs, 'snr': round(info['snr'], 2),
            'accepted': bool(info['snr'] >= args.min_snr), 'heliocentric_correction': metadata.barycorr_kms,
            'input_header_correction':header_correction, 'correction_applied':info['barycorr_applied_now'],
            'input_frame':'barycentric' if args.instrument=='HERMES' else 'observer',
            'frame_note':'HERMES input already corrected per data_one/reference.txt; only frame difference applied' if args.instrument=='HERMES' else 'FIES VHELIO applied'}
    if item['accepted']:
        peaks, ccf = pipeline.measure_ccf(spectrum)
        item.update(velocity=ccf['x'].round(3).tolist(), ccf=ccf['y'].round(6).tolist(), peaks=peaks)
    rows.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'target':args.target,'frame':'heliocentric',
        'mask':'HARPS/SOPHIE M5', 'wavelength_nm':[400,687], 'min_snr':args.min_snr,'epochs':rows}, indent=2)+'\n')
    print(path.name, item['snr'], item['accepted'], item.get('peaks', []), flush=True)
# A classic script works both on static hosting and in a local file preview.
args.output.with_name('ccf-data.js').write_text(
    'window.hiddenCompanionsData = ' + args.output.read_text().strip() + ';\n')
