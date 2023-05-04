import torch
import torch.nn as nn
import torch.nn.functional as F
from models.googlenet import BasicConv2d

relu = 0
conv2d = 1
basicConv = 2
maxpool = 3


class ExecutionUnit:

    def __init__(self, required_input: tuple, operator: dict, forwarding: list, device_num: int, layer_num: int):
        self.required_input = required_input
        # self.params = weights  # loaded in each edge device
        self.operator = operator  # 1st: operator_name, 2nd: operation parameters
        self.forwarding = forwarding
        self.device_num = device_num
        self.layer_num = layer_num

    def execute(self, x, weight=None) -> torch.Tensor:
        type = self.operator['type']
        try:
            if type == 'basicConv':
                out = F.conv2d(F.pad(x, pad=self.operator['padding']), weight, stride=self.operator['stride'])
                # out = F.batch_norm(out, torch.zeros(out.shape[1]), torch.ones(out.shape[1]), *self.operator['bn_args'])
                out = F.relu(out, inplace=True)
                return out
            # elif type == 'conv':  # 目前只有结合化的basicconv，暂不考虑单conv层
            #     return F.conv2d(F.pad(x, pad=self.operator['padding']), weight, stride=self.operator['stride'], padding=self.operator['padding'])
            elif type == 'maxpool':
                return F.max_pool2d(F.pad(x, pad=self.operator['padding']), self.operator['kernel_size'], stride=self.operator['stride'], padding=0, ceil_mode=self.operator['ceil_mode'])
            elif type == 'concat':  # x should be list of Tensor
                return torch.cat(x, 1)
            elif type == 'upsample':
                return F.upsample(x, scale_factor=self.operator['scale_factor'])
            else:
                return x
        except Exception as e:
            print(e)

    # def execute_with_args(self, args: list):
    #     if len(args) > 1:
    #         return self.execute()
