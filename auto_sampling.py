import numpy as np

from sklearn.cluster import DBSCAN
from read_file import read_events
from psee_loader import PSEELoader
import my_cython_code


def sampling_ncaltech101(path):

    fifo_length = 200
    x, y, p, t, _ = read_events(path)

    iter_num = 6
    alpha = 1.8
    cite = 2.3
    n = [0] * iter_num
    delta_t = [0] * iter_num

    total_event_num_in_label_location = []
    for i in range(iter_num):
        total_event_num_in_label_location.append(int(np.sum(t <= 40000*(i+1))))

    events_array = []
    for i in range(iter_num):
        end_index = total_event_num_in_label_location[i] - 1
        if end_index >= 5000:
            x_i = x[end_index - 5000:end_index]
            y_i = y[end_index - 5000:end_index]
            t_i = t[end_index - 5000:end_index]
            p_i = p[end_index - 5000:end_index]
        else:
            x_i = x[0:end_index]
            y_i = y[0:end_index]
            t_i = t[0:end_index]
            p_i = p[0:end_index]

        points = np.stack((x_i, y_i), axis=1)
        points = np.unique(points, axis=0)
        dbscan = DBSCAN(eps=8, min_samples=35).fit(points)
        labels = dbscan.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters_ != 0:
            S = []
            V = []
            for l in range(n_clusters_):
                cluster_x = points[labels == l, 0]
                cluster_y = points[labels == l, 1]
                D = (max(cluster_x) - min(cluster_x)) * (max(cluster_y) - min(cluster_y))

                if D == 0:
                    S.append(0)
                else:
                    S_i = D
                    S.append(S_i)
                    V.append(len(cluster_x) / (t_i[-1] - t_i[0]) / S_i)

            S = sum(S)
            V = sum(V) / (len(V) + 1e-5)

        else:
            S = (max(x_i) - min(x_i)) * (max(y_i) - min(y_i))
            V = len(t_i) / (t_i[-1] - t_i[0]) / S

        for j in range(end_index // 10):
            n[i] += 10
            begin_index = total_event_num_in_label_location[i] - 1 - 10 * j
            delta_t[i] = 40000 * (i+1) - t[begin_index]
            value = n[i] * alpha / S + delta_t[i] * V
            if value <= cite:
                continue
            else:
                break
        choose_x, choose_y = x[end_index - n[i]:end_index], y[end_index - n[i]:end_index]
        choose_p, choose_t = p[end_index - n[i]:end_index], t[end_index - n[i]:end_index]
        choose_t = (choose_t - choose_t[0]) / (choose_t[-1] - choose_t[0])

        local_sequences_lengths = np.zeros([240, 180, 2], dtype=int)
        eventnum_pre_pixel = np.zeros(n[i], dtype=int)
        eventnum_pre_pixel = my_cython_code.fill(eventnum_pre_pixel, local_sequences_lengths,
                                                 choose_x[::-1], choose_y[::-1], choose_p[::-1], n[i], fifo_length)

        choose_events = np.stack((choose_x, choose_y, choose_p, choose_t, eventnum_pre_pixel), axis=1).astype(np.float32)
        events_array.append(choose_events)

    return events_array


def sampling_gen1(path, number, temporal_aggregation_size):

    cite = 4.5
    alpha = 1.1

    interval = 250000
    fifo_length = 200
    duration = interval * temporal_aggregation_size - 1

    event = PSEELoader(path)
    annotations = PSEELoader(path.split('_td.dat')[0] + '_bbox.npy')

    annotation_count = annotations.event_count()
    total_event_num_in_label_location = []
    annotation = annotations.load_n_events(annotation_count)
    annotation_t = np.unique(annotation["t"])
    cur_time = annotation_t[number]
    event.seek_time(max(cur_time-duration, 0))
    all_events = event.load_delta_t(min(cur_time, duration))
    event.reset()
    x = np.clip(all_events["x"].astype(int), a_min=0, a_max=303)
    y = np.clip(all_events["y"].astype(int), a_min=0, a_max=239)
    p = all_events["p"].astype(int)
    t = all_events["t"].astype(int)
    label_t = []
    n = [0] * temporal_aggregation_size
    delta_t = [0] * temporal_aggregation_size

    for i in range(temporal_aggregation_size):
        event.seek_time(max(cur_time - duration, 0))
        count = len(event.load_delta_t(min(cur_time, duration)-interval*i)) if min(cur_time, duration)-interval*i > 0 else 0
        total_event_num_in_label_location.append(count)
        label_t.append(max(cur_time-interval*i, 0))
        event.reset()

    total_event_num_in_label_location.reverse()
    label_t.reverse()
    events_array = []

    for i in range(temporal_aggregation_size):
        end_index = total_event_num_in_label_location[i] - 1
        if end_index >= 5000:
            x_i = x[end_index - 5000:end_index]
            y_i = y[end_index - 5000:end_index]
            t_i = t[end_index - 5000:end_index]
        elif end_index <= 0:
            events_array.append(np.zeros([0, 5], dtype=np.float32))
            continue
        else:
            x_i = x[0:end_index]
            y_i = y[0:end_index]
            t_i = t[0:end_index]

        points = np.stack((x_i, y_i), axis=1)
        points = np.unique(points, axis=0)
        dbscan = DBSCAN(eps=12, min_samples=30).fit(points)
        labels = dbscan.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters_ != 0:
            S = []
            V = []
            for l in range(n_clusters_):
                cluster_x = points[labels == l, 0]
                cluster_y = points[labels == l, 1]
                H = max(cluster_y) - min(cluster_y)
                W = max(cluster_x) - min(cluster_x)
                D = H * W

                if D == 0:
                    S.append(0)
                else:
                    S_i = D
                    S.append(S_i)
                    V.append(len(cluster_x) / (t_i[-1] - t_i[0]) / S_i)

            S = sum(S)
            V = sum(V) / (len(V) + 1e-5)

        else:
            S = (max(x_i) - min(x_i)) * (max(y_i) - min(y_i))
            V = len(t_i) / (t_i[-1] - t_i[0]) / S

        label_t[i] = t[t < label_t[i]][-1]

        if S == 0:
            n[i] = total_event_num_in_label_location[i] - 1
        else:
            for j in range(end_index // 10):
                n[i] += 10
                begin_index = total_event_num_in_label_location[i] - 1 - 10 * j
                delta_t[i] = label_t[i] - t[begin_index]
                np.seterr(divide='ignore', invalid='ignore')

                value = alpha * n[i] / S + delta_t[i] * V

                if n[i] >= 100000 or delta_t[i] >= 500000:
                    break
                elif value <= cite:
                    continue
                else:
                    break

        choose_x, choose_y = x[end_index - n[i]:end_index], y[end_index - n[i]:end_index]
        choose_p, choose_t = p[end_index - n[i]:end_index], t[end_index - n[i]:end_index]
        choose_t = (choose_t - choose_t[0]) / (choose_t[-1] - choose_t[0])

        local_sequences_lengths = np.zeros([304, 240, 2], dtype=int)
        eventnum_pre_pixel = np.zeros(n[i], dtype=int)

        eventnum_pre_pixel = my_cython_code.fill(eventnum_pre_pixel, local_sequences_lengths,
                                                 choose_x[::-1], choose_y[::-1], choose_p[::-1], n[i], fifo_length)

        choose_events = np.stack((choose_x, choose_y, choose_p, choose_t, eventnum_pre_pixel), axis=1).astype(np.float32)
        events_array.append(choose_events)

    out = []
    for label in annotation:
        if label[0] == annotation_t[number]:
            if label[3] >= 10 and label[4] >= 10 and (label[3]**2 + label[4]**2) >= 900:
                out.append(np.array([label[1], label[2], label[3], label[4], label[0], label[5]]))
    np_bbox = np.stack(out, axis=0) if out else np.zeros([0, 6])

    array_width = np.ones_like(np_bbox[:, 0]) * 304 - 1
    array_height = np.ones_like(np_bbox[:, 1]) * 240 - 1

    np_bbox[:, :2] = np.maximum(np_bbox[:, :2], np.zeros_like(np_bbox[:, :2]))
    np_bbox[:, 0] = np.minimum(np_bbox[:, 0], array_width)
    np_bbox[:, 1] = np.minimum(np_bbox[:, 1], array_height)

    np_bbox[:, 2] = np.minimum(np_bbox[:, 2], array_width - np_bbox[:, 0])
    np_bbox[:, 3] = np.minimum(np_bbox[:, 3], array_height - np_bbox[:, 1])
    np_bbox[:, :2] = np_bbox[:, :2] + np_bbox[:, 2:4] / 2

    return events_array, np_bbox

