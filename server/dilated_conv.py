# ================================================================================
# dilated_conv.py
#
# gfx1103（Radeon 780M）では dilation 付き Conv1d が通常の畳み込みより
# 一桁以上遅い。タップの間に 0 を入れた密なカーネルへ展開して実行する。
# ================================================================================

import torch.nn.functional as F

from server.gpu_runtime import gpu_arch_name


_INSTALLED = False


# ================================================================================
# _dense_weight
# dilation 分だけ 0 を挟んだカーネルを返す。推論中は重みが変わらないので再利用する。
# ================================================================================
def _dense_weight(module, dilation):
    weight = module.weight
    width = (weight.shape[-1] - 1) * dilation + 1
    cached = getattr(module, "_irodori_dense_kernel", None)

    if (
        cached is not None
        and cached.shape == (weight.shape[0], weight.shape[1], width)
        and cached.dtype == weight.dtype
        and cached.device == weight.device
    ):
        return cached

    expanded = weight.new_zeros(weight.shape[0], weight.shape[1], width)
    expanded[..., ::dilation] = weight
    module._irodori_dense_kernel = expanded
    return expanded


# ================================================================================
# install_dense_dilated_conv
# 選択中の GPU が gfx1103 のとき、NormConv1d の dilation>1 を密な畳み込みにする。
# ================================================================================
def install_dense_dilated_conv(torch_module, device_map):
    global _INSTALLED

    if _INSTALLED:
        return True

    device_map = str(device_map or "")

    if not device_map.startswith("cuda:"):
        return False

    index = int(device_map.split(":", 1)[1])
    arch = gpu_arch_name(
        getattr(torch_module.cuda.get_device_properties(index), "gcnArchName", "")
    )

    if arch != "gfx1103":
        return False

    from dacvae.nn.layers import NormConv1d

    original_forward = NormConv1d.forward

    def forward(self, x):
        dilation = int(self.dilation[0])

        if dilation <= 1 or int(self.groups) != 1:
            return original_forward(self, x)

        x = self.pad(x)
        return F.conv1d(
            x,
            _dense_weight(self, dilation),
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=self.groups,
        )

    NormConv1d.forward = forward
    _INSTALLED = True
    print("gfx1103: dilated Conv1d を密なカーネルで実行します")
    return True


# ================================================================================
