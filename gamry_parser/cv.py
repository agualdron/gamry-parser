import gamry_parser as parser
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

class CyclicVoltammetry(parser.GamryParser):
    """Load a Cyclic Voltammetry experiment generated in Gamry EXPLAIN format."""

    def get_v_range(self):
        """ retrieve the programmed voltage scan ranges

        Args:
            None

        Returns:
            tuple, containing:
                float: voltage limit 1, in V
                float: voltage limit 2, in V

        """
        assert self.loaded, 'DTA file not loaded. Run CyclicVoltammetry.load()'
        assert 'VLIMIT1' in self.header.keys(), 'DTA header file missing VLIMIT1 specification'
        assert 'VLIMIT2' in self.header.keys(), 'DTA header file missing VLIMIT2 specification'

        return self.header['VLIMIT1'], self.header['VLIMIT2']

    def get_scan_rate(self):
        """ retrieve the programmed scan rate

        Args:
            None

        Returns:
            float: the scan rate, in mV/s

        """
        assert self.loaded, 'DTA file not loaded. Run CyclicVoltammetry.load()'
        assert 'SCANRATE' in self.header.keys(), 'DTA header file missing SCANRATE specification'
        return self.header['SCANRATE']

    def get_curve_data(self, curve=0):
        """ retrieve relevant cyclic voltammetry experimental data

        Args:
            curve (int, optional): curve number to return. Defaults to 0.

        Returns:
            pandas.DataFrame:
                - Vf: potential, in V
                - Im: current, in A

        """
        assert self.loaded, 'DTA file not loaded. Run CyclicVoltammetry.load()'
        assert curve >= 0, 'Invalid curve ({}). Indexing starts at 0'.format(
            curve)
        assert curve < self.curve_count, f'Invalid curve ({curve}). File contains {self.curve_count} total curves.'
        df = self.curves[curve]

        return df[['Vf', 'Im']]

    def get_peaks(self, curves=None, width_V=0.02, peak_range_V=None):
        """ retrieve local peaks from cyclic voltammetry curve
        See scipy.signal.find_peaks for more details. It would be useful to pass additional arguments to find_peaks.
        Currently, the "peak" at the positive voltage limit is returned. This can be prevented with peak_range_x.
        The "peak" at the negative voltage limit is not returned.
        
        Args:
            curves:         curve indexes number to find peaks. Defaults to all. A single index should be passed as a list/array
            width_V:        mininum peak width in volts. Defaults to 0.02 V = 20 mV
            peak_range_V:   peaks outside of this range will be discarded. Two values required, e.g. [-1, 1]
        
        Returns:
            pandas.DataFrame:
                - peak_heights
                - prominences
                - left_bases
                - right_bases
                - widths
                - width_heights
                - left_ips
                - right_ips
                - peak_locations
        """
        
        all_curves = range(self.get_curve_count())
        if curves is None:
            curves = all_curves
        else:
            for p in curves:
                assert p in all_curves, f'Cycle index {p} invalid'
        
        peaks = dict()
        for curve in curves:
            data = self.get_curve_data(curve)

            V = data['Vf'] # should be in V
            j = data['Im'] # should be in A
            
            # define the peak width in terms of samples
            width = int(width_V / self.get_scan_rate())
            
            # run scipy.signal.find_peaks in both directions
            pks_pos, props_pos = find_peaks(j, width=width, height=0)
            pks_neg, props_neg = find_peaks(-j, width=width, height=0)
            props_neg['peak_heights'] = -props_neg['peak_heights']
            props_neg['prominences'] = -props_neg['prominences']
            
            # merge positive + negative peaks
            pks = np.append(pks_pos, pks_neg)
            curve_peaks = dict()
            for key in props_pos.keys():
                curve_peaks[key] = np.append(props_pos[key],props_neg[key])

            # add keys for indexing
            curve_peaks["peak_locations"] = V.iloc[pks]
            curve_peaks = pd.DataFrame(curve_peaks)

            # remove unwanted peaks
            if peak_range_V is not None:
                mask = curve_peaks["peak_locations"] > peak_range_V[0] and curve_peaks["Vf"] < peak_range_V[1]
                curve_peaks = curve_peaks[mask]

            if len(peaks)==0:
                peaks = curve_peaks
            else:
                peaks = peaks.append(curve_peaks)
        
        return peaks