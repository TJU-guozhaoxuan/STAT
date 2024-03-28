import torch.nn as nn

from head import YOLOXHead
from backbone import Backbone
from neck import YOLOXPAFPN


class network(nn.Module):

    def __init__(self, classes_num, batch_size, temporal_aggregation_size, width=1.0,
                 strides=(8, 16, 32), in_channels=(128, 256, 512), act="silu", focalloss=True,
                 image_shape=(180, 240), input_shape=(224, 224)):
        super().__init__()
        self.backbone = Backbone(classes_num, batch_size, temporal_aggregation_size,
                                 input_shape=input_shape, image_shape=image_shape)
        self.neck = YOLOXPAFPN(in_channels, act)
        self.head = YOLOXHead(classes_num, temporal_aggregation_size, width, strides, in_channels, act, focalloss)

    def forward(self, x, labels):
        x, image, ev_rep_time, bone_time = self.backbone(x)
        x = self.neck(x)
        x = self.head(x, labels)
        return x, ev_rep_time, bone_time
