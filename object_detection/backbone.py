import time
import torch.nn as nn
import torch.nn.functional as F
from event_gird import Gird, MLP, RNN
import sys

from TVT import TVT
sys.path.append("..")


class Backbone(nn.Module):

    def __init__(self, classes_num, batch_size, temporal_aggregation_size,
                 image_shape=(180, 240), input_shape=(224, 224)):

        nn.Module.__init__(self)
        self.event_surface = Gird(MLP, RNN, temporal_aggregation_size=temporal_aggregation_size, input_shape=image_shape)
        self.input_shape = input_shape
        self.backbone = TVT(batch_size=batch_size, pretrained=True, drop_rate=0.5,
                            drop_path_rate=0.8, temporal_aggregation_size=temporal_aggregation_size,
                            num_classes=classes_num, in_chans=2)

    def forward(self, x):
        time0 = time.time()
        x = self.event_surface(x)
        ev_rep_time = time.time() - time0
        vox_cropped = F.interpolate(x, size=self.input_shape, mode="bilinear")
        multiscale_feature = self.backbone(vox_cropped)
        return multiscale_feature[1:], x, ev_rep_time, time.time()-time0-ev_rep_time
