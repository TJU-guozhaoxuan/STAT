import glob
import numpy as np
import sys

from torch.utils.data import Dataset
from bisect import bisect
from psee_loader import PSEELoader
from auto_sampling import sampling_ncaltech101, sampling_gen1

sys.path.append("..")


def random_shift_events(events, bounding_box, max_shift=20, resolution=(180, 240), p=0.75):

    if np.random.random() < p:

        H, W = resolution
        x_min = np.min(bounding_box[:, 0] - bounding_box[:, 2] / 2) if bounding_box.shape[0] != 0 else resolution[1]
        x_max = np.max(bounding_box[:, 0] + bounding_box[:, 2] / 2) if bounding_box.shape[0] != 0 else 0
        y_min = np.min(bounding_box[:, 1] - bounding_box[:, 3] / 2) if bounding_box.shape[0] != 0 else resolution[0]
        y_max = np.max(bounding_box[:, 1] + bounding_box[:, 3] / 2) if bounding_box.shape[0] != 0 else 0
        x_shift = np.random.randint(-min(x_min, max_shift), min(W - x_max, max_shift), size=(1,))
        y_shift = np.random.randint(-min(y_min, max_shift), min(H - y_max, max_shift), size=(1,))

        bounding_box[:, 0] += x_shift
        bounding_box[:, 1] += y_shift

        events[:, 0] += x_shift
        events[:, 1] += y_shift

        valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
        events = events[valid_events]

    return events, bounding_box


def random_flip_events_along_x(events, bounding_box, resolution=(180, 240), p=0.5):
    H, W = resolution
    flipped = False
    if np.random.random() < p:
        events[:, 0] = W - 1 - events[:, 0]
        flipped = True

    if flipped:
        bounding_box[:, 0] = W - 1 - bounding_box[:, 0]
    return events, bounding_box


class my_dataset(Dataset):

    def __init__(self, dataset_path, dataset_type="gen1", temporal_aggregation_size=3,
                 type="train", label_shape=(180, 240), adjust_shape=(224, 224), augmentation=False):
        super().__init__()
        self.files = []
        self.labels = []
        self.augmentation = augmentation
        self.dataset_type = dataset_type
        self.label_shape = label_shape
        self.temporal_aggregation_size = temporal_aggregation_size
        self.ratio_h = adjust_shape[0] / label_shape[0]
        self.ratio_w = adjust_shape[1] / label_shape[1]
        if self.dataset_type == "n-caltech101":
            if type == "train":
                paths = glob.glob(dataset_path + "/train/*.bin")
                for path in paths:
                    self.files.append(path)
                    self.labels.append(dataset_path + "/train/annotations/" + \
                                       path.split("/")[-1].split(".")[0] + ".npy")
            elif type == "val":
                paths = glob.glob(dataset_path + "/validation/*.bin")
                for path in paths:
                    self.files.append(path)
                    self.labels.append(dataset_path + "/validation/annotations/" + \
                                       path.split("/")[-1].split(".")[0] + ".npy")

            elif type == "test":
                paths = glob.glob(dataset_path + "/test/*.bin")
                for path in paths:
                    self.files.append(path)
                    self.labels.append(dataset_path + "/test/annotations/" + \
                                       path.split("/")[-1].split(".")[0] + ".npy")

            else:
                raise ValueError("type must be train, val or test")

        elif self.dataset_type == "gen1":
            if type == "train":
                paths = glob.glob(dataset_path + "/train/*.dat")

            elif type == "val":
                paths = glob.glob(dataset_path + "/val/*.dat")

            elif type == "test":
                paths = glob.glob(dataset_path + "/test/*.dat")

            else:
                raise ValueError("type must be one of train, val and test")

            self.length = 0
            self.length_list = []
            for path in paths:
                self.files.append(path)
                annotations = PSEELoader(path.split('_td.dat')[0] + '_bbox.npy')
                annotation_count = annotations.event_count()
                annotation = annotations.load_n_events(annotation_count)
                annotation_t = np.unique(annotation["t"])
                self.length += len(annotation_t)
                self.length_list.append(self.length)

        else:
            raise ValueError("dataset_type must be one of gen1 and n-caltech101")

    def __getitem__(self, idx):

        if self.dataset_type == "n-caltech101":
            label = self.labels[idx // (7 - self.temporal_aggregation_size)]
            file = self.files[idx // (7 - self.temporal_aggregation_size)]
            start_index = idx % (7 - self.temporal_aggregation_size)

            events = sampling_ncaltech101(file)[start_index:start_index+self.temporal_aggregation_size]
            annotation = np.load(label)[start_index+self.temporal_aggregation_size-1]

            array_width = np.ones_like(annotation[0]) * 240 - 1
            array_height = np.ones_like(annotation[1]) * 180 - 1

            annotation[:2] = np.maximum(annotation[:2], np.zeros_like(annotation[:2]))
            annotation[2] = np.minimum(annotation[2], array_width)
            annotation[3] = np.minimum(annotation[3], array_height)

            annotation[0], annotation[2] = (annotation[0] + annotation[2]) / 2, (
                        annotation[2] - annotation[0])
            annotation[1], annotation[3] = (annotation[1] + annotation[3]) / 2, (
                        annotation[3] - annotation[1])
            annotation = np.expand_dims(annotation, axis=0)

        elif self.dataset_type == "gen1":

            index = bisect(self.length_list, idx)
            file = self.files[index]
            num = (idx - self.length_list[index - 1]) if index >= 1 else idx
            events, annotation = sampling_gen1(file, num, self.temporal_aggregation_size)

        else:
            raise ValueError("dataset_type must be one of n-caltech101 and gen1")

        choose_events = []
        for i, event in enumerate(events):
            event = np.concatenate([event, i * np.ones((len(event), 1), dtype=np.float32)], 1)
            choose_events.append(event)
        events = np.concatenate(choose_events, axis=0)

        if self.augmentation:
            events, annotation = random_shift_events(events, annotation, resolution=self.label_shape)
            events, annotation = random_flip_events_along_x(events, annotation, resolution=self.label_shape)
        annotation[:, [0, 2]] = annotation[:, [0, 2]] * self.ratio_w
        annotation[:, [1, 3]] = annotation[:, [1, 3]] * self.ratio_h

        return events, annotation

    def __len__(self):

        if self.dataset_type == "n-caltech101":
            return len(self.files) * (7-self.temporal_aggregation_size)

        elif self.dataset_type == "gen1":
            return self.length
