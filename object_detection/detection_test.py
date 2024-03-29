import torch
import time
import tqdm

from network import network
from dataloader import Loader
from datasets import my_dataset
from utils import non_max_suppression, Compute_statistics, Cumpute_map


def test(batch_size, temporal_aggregation_size, dataset_type,
         dataset_path, ckpt_name, device):

    network_input_shape = (224, 224)

    ckpt_path = ("ckpt_ncaltech101" + "/" + dataset_type + "-" + ckpt_name) if dataset_type == "n-caltech101" \
        else ("ckpt_gen1" + "/" + dataset_type + "-" + ckpt_name)

    class_num = 100 if dataset_type == "n-caltech101" else 2
    event_surface_shape = (180, 240) if dataset_type == "n-caltech101" else (240, 304)

    test_dataset = my_dataset(dataset_path, temporal_aggregation_size=temporal_aggregation_size,
                              dataset_type=dataset_type, label_shape=event_surface_shape,
                              adjust_shape=network_input_shape, type="test")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    test_loader = Loader(test_dataset, dataset_type, device=device, batch_size=batch_size, shuffle=False)

    model = network(class_num, batch_size, temporal_aggregation_size,
                    image_shape=event_surface_shape, input_shape=network_input_shape)

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    model = model.to(device)
    model = model.eval()

    times1, times2, times3 = [], [], []
    stats = []
    for i, (events, labels) in enumerate(tqdm.tqdm(test_loader, ncols=80)):

        with torch.no_grad():
            time1 = time.time()
            pred, ev_rep_time, bone_time = model(events, labels)

        times1.append(time.time()-time1)
        times2.append(ev_rep_time)
        times3.append(bone_time)
        output = non_max_suppression(pred)
        stats = Compute_statistics(output, labels, device, stats)

    print("event representation time: ", sum(times2)/len(times2))
    print("network backbone time: ", sum(times3) / len(times3))
    print("inference time: ", sum(times1) / len(times1))
    Cumpute_map(stats)


if __name__ == '__main__':
    batch_size = 5
    temporal_aggregation_size = 3
    dataset_type = "n-caltech101"                           # "gen1" or "n-caltech101"
    dataset_path = "../data/" + \
                   "ncaltech101" if dataset_type == "n-caltech101" else "detection_dataset_duration_60s_ratio_1.0"
    ckpt_name = "best.pth"
    device = "cpu"
    test(batch_size, temporal_aggregation_size, dataset_type,
         dataset_path, ckpt_name, device)
