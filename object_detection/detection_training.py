import torch
import os
import tqdm
import math

from network import network
from torch.utils.tensorboard import SummaryWriter
from dataloader import Loader
from datasets import my_dataset
from utils import non_max_suppression, Compute_statistics, Cumpute_map


def percentile(t, q):
    B, C, H, W = t.shape
    k = 1 + round(.01 * float(q) * (C * H * W - 1))
    result = t.view(B, -1).kthvalue(k).values
    return result[:,None,None,None]


def getModelSize(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    all_size = (param_size * 4 + buffer_size) / 1024 / 1024
    print('Parameter number: {:.3f}MB'.format(param_size/1024/1024))
    print('Model size：{:.3f}MB'.format(all_size))


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train(batch_size, temporal_aggregation_size, epochs, data_augmentation,
          dataset_type, dataset_path, best_ckpt_name, last_ckpt_name, device):

    warm_up_iter = 3
    T_max = 10
    lr_max = 1e-4
    lr_min = 1e-6

    warm_up = False
    use_ema = True
    clip_grad = True
    focal_loss = True
    gradient_accumulation = False

    if gradient_accumulation:
        accumulation_steps = 3
        lr_max = lr_max * accumulation_steps
        lr_min = lr_min * accumulation_steps

    if clip_grad:
        max_grad = 3

    event_surface_shape = (180, 240) if dataset_type == "n-caltech101" else (240, 304)
    nework_input_shape = (224, 224)
    log_dir = "log_dir_ncaltech101" if dataset_type == "n-caltech101" else "log_dir_gen1"
    ckpt_dir = "ckpt_ncaltech101" if dataset_type == "n-caltech101" else "ckpt_gen1"

    classes_num = 100 if dataset_type == "n-caltech101" else 2

    training_dataset = my_dataset(dataset_path, temporal_aggregation_size=temporal_aggregation_size,
                                  augmentation=data_augmentation, dataset_type=dataset_type, label_shape=event_surface_shape,
                                  adjust_shape=nework_input_shape, type="train")
    val_dataset = my_dataset(dataset_path, temporal_aggregation_size=temporal_aggregation_size,
                             dataset_type=dataset_type, label_shape=event_surface_shape,
                             adjust_shape=nework_input_shape, type="val")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    training_loader = Loader(training_dataset, dataset_type, device, batch_size=batch_size)
    val_loader = Loader(val_dataset, dataset_type, device, batch_size=batch_size)

    model = network(classes_num, batch_size, temporal_aggregation_size, focalloss=focal_loss,
                    image_shape=event_surface_shape, input_shape=nework_input_shape)

    model = model.to(device)
    getModelSize(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=lr_min)

    warm_up_iter, t_max = warm_up_iter * len(training_loader), T_max * len(training_loader)
    lr_lambda = lambda cur_iter: cur_iter / warm_up_iter if cur_iter < warm_up_iter else \
        (lr_min + 0.5 * (lr_max - lr_min) * (
                    1.0 + math.cos((cur_iter - warm_up_iter) / (t_max - warm_up_iter) * math.pi))) / lr_max

    if os.path.exists(ckpt_dir + "/" + last_ckpt_name):
        checkpoint = torch.load(ckpt_dir + "/" + last_ckpt_name, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint["best-map"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"The model was successfully loaded, and training began from epoch {start_epoch}. The current best_map is {best_map}.")
    else:
        start_epoch = 0
        best_map = 0
        print("No saved model, training from scratch.")

    if warm_up:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    ema = EMA(model, 0.999)
    ema.register()

    writer = SummaryWriter(log_dir=log_dir)

    iteration = 0
    for i in range(start_epoch, epochs):

        sum_loss = 0
        model = model.train()
        print(f"Training step [{i + 1:3d}/{epochs:3d}]")
        for j, (events, labels) in enumerate(tqdm.tqdm(training_loader, ncols=80)):

            if not gradient_accumulation:
                optimizer.zero_grad()

            loss, _, _ = model(events, labels)

            if gradient_accumulation:
                loss = loss / accumulation_steps

            if use_ema:
                ema.apply_shadow()

            loss.backward()

            if gradient_accumulation:
                if ((j + 1) % accumulation_steps) == 0:
                    if clip_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)

                optimizer.step()

            if i % 3 == 2 and use_ema:
                ema.update()

            sum_loss += (loss.item() * accumulation_steps) if gradient_accumulation else loss.item()
            iteration += 1
            if warm_up:
                lr_scheduler.step()

        training_loss = sum_loss / len(training_loader)
        print(f"Training Loss {training_loss:.4f}  learning rate {optimizer.param_groups[0]['lr']}")

        if not warm_up:
            lr_scheduler.step()

        writer.add_scalar("training/loss", training_loss, i+1)

        # representation_vizualization = create_image(representation)
        # writer.add_image("training/representation", representation_vizualization, iteration)

        model = model.eval()

        print(f"Validation step [{i + 1:3d}/{epochs:3d}]")
        stats = []

        for events, labels in tqdm.tqdm(val_loader, ncols=80):
            with torch.no_grad():
                pred, _, _ = model(events, labels)

            output = non_max_suppression(pred)
            stats = Compute_statistics(output, labels, device, stats)

        map50, map75, map = Cumpute_map(stats)

        is_better = map > best_map
        best_map = map if is_better else best_map
        state_dict = model.state_dict()

        if is_better:
            torch.save({
                "state_dict": state_dict,
                "epoch": i + 1, "best-map": best_map,
                "scheduler": lr_scheduler.state_dict(),
                "optimizer": optimizer.state_dict()}, ckpt_dir + "/" + best_ckpt_name)

        torch.save({
            "state_dict": state_dict,
            "epoch": i + 1, "best-map": best_map,
            "scheduler": lr_scheduler.state_dict(),
            "optimizer": optimizer.state_dict()}, ckpt_dir + "/" + last_ckpt_name)

        writer.add_scalar("val/map50", map50, i+1)
        # writer.add_scalar("val/map75", map75, i + 1)
        writer.add_scalar("val/map", map, i+1)
        writer.add_scalar("val/best_map", best_map, i+1)
        print(f"best map is : {best_map:.4f}%")
        # representation_vizualization = create_image(representation)
        # writer.add_image("val/representation", representation_vizualization, iteration


if __name__ == '__main__':
    batch_size = 5
    temporal_aggregation_size = 3
    epochs = 60
    dataset_type = "gen1"                            # "gen1" or "n-caltech101"
    dataset_path = "../data/" + \
                   "ncaltech101" if dataset_type == "n-caltech101" else "detection_dataset_duration_60s_ratio_1.0"
    best_ckpt_name = dataset_type + "-best.pth"
    last_ckpt_name = dataset_type + "-last.pth"
    device = "cuda:0"
    train(batch_size, temporal_aggregation_size, epochs, True,
          dataset_type, dataset_path, best_ckpt_name, last_ckpt_name, device)

