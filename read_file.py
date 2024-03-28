import numpy as np


def read_events(path):
    a = np.fromfile(path, dtype=np.uint8)
    a = np.uint32(a)
    all_y = a[1::5]
    all_x = a[0::5]
    all_p = (a[2::5] & 128) >> 7
    all_ts = ((a[2::5] & 127) << 16) | (a[3::5] << 8) | (a[4::5])
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment
    td_indices = np.where(all_y != 240)[0]
    x = np.array(all_x[td_indices], dtype=np.int64)
    y = np.array(all_y[td_indices], dtype=np.int64)
    ts = np.array(all_ts[td_indices], dtype=np.int32)
    p = np.array(all_p[td_indices], dtype=np.int64)
    length = len(x)
    return x, y, p, ts, length
