import numpy as np
from sklearn.cluster import DBSCAN
cimport numpy as np


cpdef fill(np.ndarray[long, ndim=1] eventnum_pre_pixel,
           np.ndarray[long, ndim=3] local_sequences_lengths,
           np.ndarray[long, ndim=1] choose_x,
           np.ndarray[long, ndim=1] choose_y,
           np.ndarray[long, ndim=1] choose_p,
           int num, int fifo_length):

    for k in range(num):
        local_sequences_lengths[choose_x[k], choose_y[k], choose_p[k]] += 1
        eventnum_pre_pixel[k] = local_sequences_lengths[choose_x[k], choose_y[k], choose_p[k]]

    eventnum_pre_pixel = eventnum_pre_pixel[::-1]
    eventnum_pre_pixel = np.clip(eventnum_pre_pixel, 1, fifo_length)
    local_sequences_lengths = np.clip(local_sequences_lengths, 1, fifo_length)

    for k in range(num):
        eventnum_pre_pixel[k] = local_sequences_lengths[choose_x[::-1][k], choose_y[::-1][k], choose_p[::-1][k]] - eventnum_pre_pixel[k] + 1

    return eventnum_pre_pixel


cpdef auto_sampling(cite, alpha, fifo_length, x, y, p, t, n, delta_t, temporal_aggregation_size,
                    total_event_num_in_label_location, label_t, events_array):

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
        dbscan = DBSCAN(eps=8, min_samples=30).fit(points)
        labels = dbscan.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters_ != 0:
            S = []
            for l in range(n_clusters_):
                cluster_x = points[labels == l, 0]
                cluster_y = points[labels == l, 1]
                x_mean = sum(cluster_x) / len(cluster_x)
                y_mean = sum(cluster_y) / len(cluster_y)

                D = (max(cluster_x) - min(cluster_x)) * (max(cluster_y) - min(cluster_y))
                dis = np.power(np.power(cluster_x - x_mean, 2) + np.power(cluster_y - y_mean, 2), 0.5)
                compactness = sum(dis) / len(dis)
                if D == 0:
                    S.append(0)
                else:
                    S.append(D / compactness)
            V = len(t_i) / (t_i[-1] - t_i[0])
            S = sum(S)

        else:
            x_mean = sum(x_i) / len(x_i)
            y_mean = sum(y_i) / len(y_i)

            dis = np.power(np.power(x_i - x_mean, 2) + np.power(y_i - y_mean, 2), 0.5)
            D = (max(x_i) - min(x_i)) * (max(y_i) - min(y_i))
            compactness = sum(dis) / len(dis)

            V = len(t_i) / (t_i[-1] - t_i[0])
            S = D / compactness

        label_t[i] = t[t < label_t[i]][-1]

        if S == 0:
            n[i] = total_event_num_in_label_location[i] - 1
        else:
            for j in range(end_index // 10):
                n[i] += 10
                begin_index = total_event_num_in_label_location[i] - 1 - 10 * j
                delta_t[i] = label_t[i] - t[begin_index]
                np.seterr(divide='ignore', invalid='ignore')

                value = alpha * n[i] / S + delta_t[i] * V / S

                if n[i] >= 100000 or delta_t[i] >= 400000:
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

        eventnum_pre_pixel = fill(eventnum_pre_pixel, local_sequences_lengths,
                                  choose_x[::-1], choose_y[::-1], choose_p[::-1], n[i], fifo_length)

        choose_events = np.stack((choose_x, choose_y, choose_p, choose_t, eventnum_pre_pixel), axis=1).astype(np.float32)
        events_array.append(choose_events)
    return events_array
