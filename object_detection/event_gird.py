import random
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence


class MLP(nn.Module):

    def __init__(self, input_channel, hidden_channel, output_channel, clip_num=5):
        super().__init__()
        self.clip_num = clip_num
        self.hidden_layer1 = nn.Linear(input_channel, hidden_channel)
        self.hidden_layer2 = nn.Linear(hidden_channel, hidden_channel)
        self.output_layer = nn.Linear(hidden_channel, output_channel)
        self.init_kernel()

    def forward(self, x):
        list_of_input = torch.chunk(x, dim=0, chunks=self.clip_num)
        outs = []
        for i in range(min(self.clip_num, len(list_of_input))):
            input = list_of_input[i]
            out = self.hidden_layer1(input)
            out = F.leaky_relu(out, 0.1)
            out = self.hidden_layer2(out)
            out = F.leaky_relu(out, 0.1)
            out = self.output_layer(out)
            outs.append(out)
        return torch.cat(outs, dim=0)

    def init_kernel(self):
        ts = torch.zeros((2000, 1))
        optim = torch.optim.Adam(self.parameters(), lr=5e-3)

        for _ in range(1000):

            optim.zero_grad()
            ts.uniform_(-1, 1)

            gt_values = self.trilinear_kernel(ts)
            values = self.forward(ts)

            loss = (values - gt_values).pow(2).sum()
            loss.backward()
            optim.step()

    @staticmethod
    def trilinear_kernel(ts):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - ts)[ts > 0]
        gt_values[ts < 0] = (ts + 1)[ts < 0]

        gt_values[ts < -1.0] = 0
        gt_values[ts > 1.0] = 0

        return gt_values


class RNN(nn.Module):

    def __init__(self, inout_channel, hidden_channel, clip_num=5):
        super().__init__()
        self.hidden_channel = hidden_channel
        self.clip_num = clip_num
        self.layer1 = nn.Linear(inout_channel, hidden_channel)
        self.layer2 = nn.Linear(hidden_channel, inout_channel)
        self.lstm = nn.LSTM(hidden_channel, hidden_channel, num_layers=1, batch_first=True)
        # self.init_kernel()

    def forward(self, x, length):

        list_of_input = torch.chunk(x, dim=0, chunks=self.clip_num)
        list_of_length = torch.chunk(length, dim=0, chunks=self.clip_num)
        outs = []
        self.lstm.flatten_parameters()
        for i in range(min(self.clip_num, len(list_of_input))):
            short_x = self.layer1(list_of_input[i])
            packed_input = pack_padded_sequence(input=short_x, lengths=list_of_length[i], batch_first=True, enforce_sorted=False)
            _, (ht, ct) = self.lstm(packed_input)
            out = self.layer2(ht).squeeze(0)
            out = out.squeeze(-1)
            outs.append(out)
        return torch.cat(outs, dim=0)

    def init_kernel(self):
        optim = torch.optim.Adam(self.parameters(), lr=5e-3)

        for _ in range(1000):
            optim.zero_grad()

            x = torch.zeros((100, random.randint(30, 500), 1))
            length = torch.randint(1, x.shape[1], (100,))
            for i in range(len(x)):
                x[i, 0:length[i]].uniform_(-1, 1)

            values = self.forward(x, length)
            gt_values = x.squeeze().sum(dim=1)

            loss = (values - gt_values).pow(2).sum()
            loss.backward()
            optim.step()


class Gird(nn.Module):

    def __init__(self, MLP, RNN, temporal_aggregation_size=3,
                 mlp_hiddenchannel=20, rnn_hiddenchannel=20, input_shape=(180, 240)):
        super().__init__()
        self.mlp = MLP(1, mlp_hiddenchannel, 1)
        self.dim = input_shape
        self.rnn = RNN(1, rnn_hiddenchannel)
        self.temporal_aggregation_size = temporal_aggregation_size

    def forward(self, events):
        events = events.to(torch.float32)
        H, W = self.dim

        B = int((1 + events[-1, -1]).item()) * int((1 + events[-1, -2]).item())
        x, y, p, t, index_in_pixel, index_in_temporal, batch = events.t()
        x, y, p, index_in_pixel, index_in_temporal, batch = \
            x.long(), y.long(), p.long(), index_in_pixel.long(), index_in_temporal.long(), batch.long()

        max_event_num_in_pixel = torch.max(index_in_pixel)
        vox = events[0].new_full([B * 2 * H * W, max_event_num_in_pixel], fill_value=0)
        values = self.mlp.forward(t.unsqueeze(1))
        idx = index_in_pixel - 1 + max_event_num_in_pixel * x + \
              max_event_num_in_pixel * W * y + max_event_num_in_pixel * W * H * p + \
              max_event_num_in_pixel * W * H * 2 * (batch * self.temporal_aggregation_size + index_in_temporal)

        vox.put_(idx.long(), values.squeeze())

        use_rnn = vox.sum(dim=1) != 0
        rnn_input = vox[use_rnn].unsqueeze(-1)
        length = (rnn_input != 0).sum(dim=1).squeeze().detach().cpu()
        rnn_output = self.rnn(rnn_input, length)

        gird = events[0].new_full([B * 2 * H * W], fill_value=0)
        gird[use_rnn] = rnn_output
        gird = gird.reshape(B, 2, H, W)

        return gird

