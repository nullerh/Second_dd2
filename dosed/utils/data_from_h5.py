"""Transform a folder with h5 files into a dataset for dosed"""

import numpy as np

import h5py

from ..preprocessing import normalizers
#from scipy.interpolate import interp1d


def get_h5_data(filename, signals, fs=128):
    with h5py.File(filename, "r") as h5:
        try:
            # For MrOS data
            picks = ["c3", "c4", "eogl", "eogr", "chin", "legl", "legr", "nasal", "abdo", "thor"]
            waveforms = h5['data']['scaled']
            sig_size = waveforms['c3'].size
            normalizer = normalizers[signals[0]['processing']["type"]](**signals[0]['processing']['args'])

            halv = 1
            if fs == 64:
                sig_size = sig_size/2
                halv = 2
            data = np.zeros((10, int(sig_size)))
            for i, pick in enumerate(picks):
                data[i, :] = normalizer(waveforms[pick][0::halv])

        except(KeyError):
            # For dosed data
            signal_size = int(fs * min(
                    set([h5[signal["h5_path"]].size / signal['fs'] for signal in signals])
                ))

            t_target = np.cumsum([1 / fs] * signal_size)
            data = np.zeros((len(signals), signal_size))

            for i, signal in enumerate(signals):
                t_source = np.cumsum([1 / signal["fs"]] *
                                         h5[signal["h5_path"]].size)
                normalizer = normalizers[signal['processing']["type"]](**signal['processing']['args'])
                data[i, :] = interp1d(t_source, normalizer(h5[signal["h5_path"]][:]),
                                          fill_value="extrapolate")(t_target)
    return data

def get_h5_events(filename, event, fs):
    with h5py.File(filename, "r") as h5:
        try:
            # For MrOS data
            starts = h5['events'][event["h5_path"]]["start"][:]
            durations = h5['events'][event["h5_path"]]["duration"][:]

            assert len(starts) == len(durations), "Inconsistents event durations and starts"
            data = np.zeros((2, len(starts)))

            data[0, :] = starts * 128
            data[1, :] = durations * 128


        except(KeyError):
            # For dosed data
            starts = h5[event["h5_path"]]["start"][:]
            durations = h5[event["h5_path"]]["duration"][:]

            assert len(starts) == len(durations), "Inconsistents event durations and starts"

            data = np.zeros((2, len(starts)))
            data[0, :] = starts * fs
            data[1, :] = durations * fs

    return data
