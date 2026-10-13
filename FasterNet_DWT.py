import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.reduced_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, self.reduced_c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.reduced_c)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(self.reduced_c, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(self.reduced_c, in_channels, kernel_size=1, bias=False)
    def forward(self, x):
        b, c, h, w = x.shape
        x_h = F.avg_pool2d(x, (1, w))
        x_w = F.avg_pool2d(x, (h, 1))
        x_h = x_h.view(b, c, h)
        x_w = x_w.view(b, c, w)
        y = torch.cat([x_h, x_w], dim=2)
        y = y.unsqueeze(-1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_h = self.conv_h(x_h)
        x_w = self.conv_w(x_w)
        x_h = torch.sigmoid(x_h)
        x_w = torch.sigmoid(x_w)

        # 加权
        out = x * x_h * x_w
        return out
class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.ll = nn.Parameter(torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]], dtype=torch.float32), requires_grad=False)
        self.lh = nn.Parameter(torch.tensor([[[[0.5, 0.5], [-0.5, -0.5]]]], dtype=torch.float32), requires_grad=False)
        self.hl = nn.Parameter(torch.tensor([[[[0.5, -0.5], [0.5, -0.5]]]], dtype=torch.float32), requires_grad=False)
        self.hh = nn.Parameter(torch.tensor([[[[0.5, -0.5], [-0.5, 0.5]]]], dtype=torch.float32), requires_grad=False)
    def forward(self, x):
        B, C, H, W = x.shape
        orig_H, orig_W = H, W
        pad_h = (H % 2)
        pad_w = (W % 2)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H, W = x.shape[-2:]
        def conv_down(filt):
            return F.conv2d(x, filt.repeat(C, 1, 1, 1), stride=2, groups=C)
        ll = conv_down(self.ll)
        lh = conv_down(self.lh)
        hl = conv_down(self.hl)
        hh = conv_down(self.hh)
        return ll, lh, hl, hh, (orig_H, orig_W)
class IDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.ll = nn.Parameter(torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]], dtype=torch.float32), requires_grad=False)
        self.lh = nn.Parameter(torch.tensor([[[[0.5, 0.5], [-0.5, -0.5]]]], dtype=torch.float32), requires_grad=False)
        self.hl = nn.Parameter(torch.tensor([[[[0.5, -0.5], [0.5, -0.5]]]], dtype=torch.float32), requires_grad=False)
        self.hh = nn.Parameter(torch.tensor([[[[0.5, -0.5], [-0.5, 0.5]]]], dtype=torch.float32), requires_grad=False)
    def forward(self, ll, lh, hl, hh, target_size=None):
        B, C, H, W = ll.shape
        def up_conv(filt, subband):
            weight = filt.repeat(C, 1, 1, 1)
            return F.conv_transpose2d(subband, weight, stride=2, groups=C)
        out = up_conv(self.ll, ll) + up_conv(self.lh, lh) + up_conv(self.hl, hl) + up_conv(self.hh, hh)
        if target_size is not None:
            out = out[:, :, :target_size[0], :target_size[1]]
        return out
class DWTBranch(nn.Module):
    def __init__(self, dim, use_lowpass_only=False):
        super().__init__()
        self.dwt = DWT()
        self.idwt = IDWT()
        self.use_lowpass_only = use_lowpass_only
        if use_lowpass_only:
            self.process_ll = nn.Conv2d(dim, dim, kernel_size=1)
        else:
            self.process_ll = nn.Conv2d(dim, dim, kernel_size=1)
            self.process_lh = nn.Conv2d(dim, dim, kernel_size=1)
            self.process_hl = nn.Conv2d(dim, dim, kernel_size=1)
            self.process_hh = nn.Conv2d(dim, dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(dim)
    def forward(self, x):
        ll, lh, hl, hh, (orig_H, orig_W) = self.dwt(x)
        if self.use_lowpass_only:
            ll_processed = self.process_ll(ll)
            lh = torch.zeros_like(lh)
            hl = torch.zeros_like(hl)
            hh = torch.zeros_like(hh)
        else:
            ll_processed = self.process_ll(ll)
            lh_processed = self.process_lh(lh)
            hl_processed = self.process_hl(hl)
            hh_processed = self.process_hh(hh)
        if self.use_lowpass_only:
            out = self.idwt(ll_processed, lh, hl, hh, target_size=(orig_H, orig_W))
        else:
            out = self.idwt(ll_processed, lh_processed, hl_processed, hh_processed, target_size=(orig_H, orig_W))
        return self.bn(out)
class PartialConv(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_partial = dim // n_div
        self.dim_untouched = dim - self.dim_partial
        self.partial_conv = nn.Conv2d(self.dim_partial, self.dim_partial,
                                      kernel_size=3, padding=1, stride=1, bias=False)
        self.forward = self.forward_slice if forward == 'slice' else self.forward_index
    def forward_slice(self, x):
        x1 = x[:, :self.dim_partial, :, :]
        x2 = x[:, self.dim_partial:, :, :]
        x1 = self.partial_conv(x1)
        return torch.cat((x1, x2), dim=1)
    def forward_index(self, x):
        idx = torch.arange(self.dim_partial, device=x.device)
        x[:, idx] = self.partial_conv(x[:, idx])
        return x
class FasterNetBlock(nn.Module):
    def __init__(self, dim, n_div=4, forward='slice', 
                 use_dwt_branch=False, dwt_lowpass_only=False,
                 use_ca=False, ca_reduction=16):
        super().__init__()
        self.pconv = PartialConv(dim, n_div, forward)
        self.use_dwt = use_dwt_branch
        if use_dwt_branch:
            self.dwt_branch = DWTBranch(dim, use_lowpass_only=dwt_lowpass_only)
            self.fuse_alpha = nn.Parameter(torch.tensor(0.5))
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=1, padding=0)
        self.act = nn.ReLU()
        self.use_ca = use_ca
        if use_ca:
            self.ca = CoordinateAttention(dim, reduction=ca_reduction)
    def forward(self, x):
        residual = x
        x_p = self.pconv(x)
        if self.use_dwt:
            x_dwt = self.dwt_branch(x)
            x = x_p + self.fuse_alpha * x_dwt
        else:
            x = x_p
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.use_ca:
            x = self.ca(x)
        return x + residual
class DynamicDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()
        self.conv_down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2,
                                   padding=dilation, dilation=dilation, bias=False)
        self.pool_down = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        )
        self.dwt = DWT()
        self.dwt_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 3),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        out_conv = self.conv_down(x)
        out_pool = self.pool_down(x)
        ll, _, _, _, (orig_H, orig_W) = self.dwt(x)
        out_dwt = self.dwt_conv(ll)
        weights = self.gate(x).view(B, 3, 1, 1)
        out = weights[:, 0:1] * out_conv + weights[:, 1:2] * out_pool + weights[:, 2:3] * out_dwt
        return out
class FasterNet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=64,
                 depths=(1, 2, 8, 2),
                 n_div=4,
                 forward='slice',
                 use_dwt_branch=True,
                 dwt_lowpass_only=False,
                 dynamic_downsample=True,
                 downsample_dilation=2,
                 use_ca=False,
                 ca_reduction=16,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)

        self.stages = nn.ModuleList()
        for i, depth in enumerate(depths):
            stage_blocks = nn.Sequential(
                *[FasterNetBlock(
                    embed_dim * (2 ** i), n_div, forward,
                    use_dwt_branch=use_dwt_branch,
                    dwt_lowpass_only=dwt_lowpass_only,
                    use_ca=use_ca,
                    ca_reduction=ca_reduction
                ) for _ in range(depth)]
            )
            self.stages.append(stage_blocks)

            if i != len(depths) - 1:
                in_ch = embed_dim * (2 ** i)
                out_ch = embed_dim * (2 ** (i + 1))
                if dynamic_downsample:
                    downsample = DynamicDownsample(in_ch, out_ch, dilation=downsample_dilation)
                else:
                    downsample = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2,
                                           padding=downsample_dilation, dilation=downsample_dilation,
                                           bias=False)
                self.stages.append(downsample)
        self.norm = nn.BatchNorm2d(embed_dim * (2 ** (len(depths) - 1)))
        self.head = nn.Linear(embed_dim * (2 ** (len(depths) - 1)), num_classes)

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1.0)
            init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
    def forward_features(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        return x
    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean([-2, -1])
        x = self.head(x)
        return x
def fasternet_tiny(num_classes=1000, use_dwt_branch=False, dwt_lowpass_only=False,
                   dynamic_downsample=True, downsample_dilation=2,
                   use_ca=False, ca_reduction=16):
    return FasterNet(embed_dim=64, depths=(1,2,8,2), num_classes=num_classes,
                     use_dwt_branch=use_dwt_branch, dwt_lowpass_only=dwt_lowpass_only,
                     dynamic_downsample=dynamic_downsample, downsample_dilation=downsample_dilation,
                     use_ca=use_ca, ca_reduction=ca_reduction)
def fasternet_small(num_classes=1000, use_dwt_branch=False, dwt_lowpass_only=False,
                    dynamic_downsample=True, downsample_dilation=2,
                    use_ca=False, ca_reduction=16):
    return FasterNet(embed_dim=128, depths=(1,2,8,2), num_classes=num_classes,
                     use_dwt_branch=use_dwt_branch, dwt_lowpass_only=dwt_lowpass_only,
                     dynamic_downsample=dynamic_downsample, downsample_dilation=downsample_dilation,
                     use_ca=use_ca, ca_reduction=ca_reduction)
def fasternet_medium(num_classes=1000, use_dwt_branch=False, dwt_lowpass_only=False,
                     dynamic_downsample=True, downsample_dilation=2,
                     use_ca=False, ca_reduction=16):
    return FasterNet(embed_dim=128, depths=(2,4,12,4), num_classes=num_classes,
                     use_dwt_branch=use_dwt_branch, dwt_lowpass_only=dwt_lowpass_only,
                     dynamic_downsample=dynamic_downsample, downsample_dilation=downsample_dilation,
                     use_ca=use_ca, ca_reduction=ca_reduction)
def fasternet_large(num_classes=1000, use_dwt_branch=False, dwt_lowpass_only=False,
                    dynamic_downsample=True, downsample_dilation=2,
                    use_ca=False, ca_reduction=16):
    return FasterNet(embed_dim=192, depths=(2,4,12,4), num_classes=num_classes,
                     use_dwt_branch=use_dwt_branch, dwt_lowpass_only=dwt_lowpass_only,
                     dynamic_downsample=dynamic_downsample, downsample_dilation=downsample_dilation,
                     use_ca=use_ca, ca_reduction=ca_reduction)