import torch as t
import torch.nn as nn

class QuanConv2d(nn.Conv2d):
    def __init__(self, m: nn.Conv2d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        x = self.quan_a_fn(x)
        return self._conv_forward(x, quantized_weight, self.bias)


class QuanAct(t.nn.Module):
    def __init__(self, act, quan_a_fn=None):
        super().__init__()
        self.act = act
        self.quan_a_fn = quan_a_fn

    def forward(self, x):
        return self.quan_a_fn(self.act(x))


