import torch
import torch.nn as nn
from torchsummary import summary

MAX_DIM = 64 * 16


def _concat(x, attr):
    n, state_dim, h, w = x.size()
    att_dim = attr.size()[1]
    attr = attr.view((n, att_dim, 1, 1)).expand((n, att_dim, h, w))
    return torch.cat([x, attr], 1)


class conv_norm_active(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2,
                 padding=1, bias=True, conv_mode='conv', norm_mode='bn',
                 act_mode='lrelu'):
        super(conv_norm_active, self).__init__()

        # conv module
        if conv_mode == 'conv':
            conv_module = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)
        elif conv_mode == 'dconv':
            conv_module = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, bias=bias)
        else:
            raise NotImplementedError

        # norm module
        if norm_mode == 'bn':
            norm_module = nn.BatchNorm2d(out_channels)
        elif norm_mode == 'in':
            norm_module = nn.InstanceNorm2d(
                out_channels, affine=True, track_running_stats=True)
        else:
            raise NotImplementedError

        # activate module
        if act_mode == 'sigmoid':
            act_module = nn.Sigmoid()
        elif act_mode == 'tanh':
            act_module = nn.Tanh()
        elif act_mode == 'lrelu':
            act_module = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif act_mode == 'relu':
            act_module = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError

        # features
        self.features = nn.Sequential(
            conv_module,
            norm_module,
            act_module
        )

    def forward(self, x):
        x = self.features(x)
        return x


class Genc(nn.Module):
    def __init__(self, dim=64, n_layers=5, multi_inputs=1):
        super(Genc, self).__init__()
        in_channels = 3
        self.multi_inputs = multi_inputs
        self.encoder = nn.ModuleList()
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            self.encoder.append(conv_norm_active(
                in_channels, d, 4, 2, 1, conv_mode='conv',
                norm_mode='bn', act_mode='lrelu'))
            in_channels = d

    def forward(self, x):
        _, c, h, w = x.size()
        zs = []
        x_ = x
        for i, layer in enumerate(self.encoder):
            if self.multi_inputs > i and i > 0:
                x_ = torch.cat([x_, torch.nn.Upsample(
                    size=(h // (2**i), w // (2**i)), mode='bicubic')(x)])
            x_ = layer(x_)
            zs.append(x_)
        return zs


class ConvGRUCell(nn.Module):
    def __init__(self, in_dim, state_dim, out_dim, kernel_size=3):
        super(ConvGRUCell, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            state_dim, out_dim, 4, 2, 1)
        self.reset_gate = conv_norm_active(
            in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2,
            conv_mode='conv', norm_mode='bn', act_mode='sigmoid')
        self.update_gate = conv_norm_active(
            in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2,
            conv_mode='conv', norm_mode='bn', act_mode='sigmoid')
        self.hidden = conv_norm_active(
            in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2,
            conv_mode='conv', norm_mode='bn', act_mode='tanh')

    def forward(self, input, state):
        n, _, h, w = state.size()
        state_ = self.upsample(state)
        r = self.reset_gate(torch.cat([input, state_], dim=1))
        z = self.update_gate(torch.cat([input, state_], dim=1))
        new_state = r * state_
        hidden_info = self.hidden(torch.cat([input, new_state], dim=1))
        output = (1 - z) * state_ + z * hidden_info
        return output, new_state


class Gstu(nn.Module):
    def __init__(self, att_dim, dim=64, enc_layers=5, n_layers=4,
                 inject_layers=4, kernel_size=3, norm='none'):
        super(Gstu, self).__init__()
        self.stu = nn.ModuleList()
        self.n_layers = n_layers
        self.inject_layers = inject_layers

        # init dims
        state_dim = att_dim + dim * 2**(enc_layers - 1)
        for i in range(n_layers):
            d = min(dim * 2**(n_layers - 1 - i), MAX_DIM)
            self.stu.append(ConvGRUCell(
                d, state_dim, d, kernel_size=kernel_size))
            if inject_layers > i:
                state_dim = d + att_dim
            else:
                state_dim = d

    def forward(self, zs, _a):
        n, state_dim, h, w = zs[-1].size()
        zs_ = [zs[-1].clone()]

        state = _concat(zs[-1], _a)
        for i, layer in enumerate(self.stu):
            output_, state_ = layer(zs[self.n_layers - 1 - i], state)
            zs_.append(output_)
            if self.inject_layers > i:
                state = _concat(state_, _a)
            else:
                state = state_
        return zs_


class Gdec(nn.Module):
    def __init__(self, att_dim, dim=64, n_layers=5, shortcut_layers=1,
                 inject_layers=0, is_training=True, one_more_conv=0):
        super(Gdec, self).__init__()
        self.decoder = nn.ModuleList()
        self.n_layers = n_layers
        self.shortcut_layers = shortcut_layers
        self.inject_layers = inject_layers

        in_dim = min(dim * 2**(n_layers - 1), MAX_DIM) + att_dim
        for i in range(n_layers):
            d = min(dim * 2**(n_layers - 1 - i), MAX_DIM)
            up_d = min(dim * 2**(n_layers - 2 - i), MAX_DIM)
            if i < n_layers - 1:
                self.decoder.append(conv_norm_active(
                    in_dim, d, 4, 2, 1, conv_mode='dconv',
                    norm_mode='bn', act_mode='relu'))

                in_dim = d
                if shortcut_layers > i:
                    in_dim += up_d

                if inject_layers > i:
                    in_dim += att_dim

            else:
                if one_more_conv:
                    self.decoder.append(nn.Sequential(
                        conv_norm_active(
                            in_dim, dim // 4, 4, 2, 1,
                            conv_mode='dconv', norm_mode='bn',
                            act_mode='relu'),
                        nn.ConvTranspose2d(dim // 4, 3, one_more_conv, 1),
                        nn.Tanh()
                    ))
                else:
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(in_dim, 3, 4, 2, 1),
                        nn.Tanh()
                    ))

    def forward(self, zs_, _a):
        z = _concat(zs_[0], _a)
        for i, layer in enumerate(self.decoder):
            z = layer(z)
            if i < self.n_layers - 1:
                if self.shortcut_layers > i:
                    z = torch.cat([z, zs_[i + 1]], dim=1)
                if self.inject_layers > i:
                    z = _concat(z, _a)
        return z


class G_stgan(nn.Module):
    def __init__(self, attr_dim, enc_dim=64, stu_dim=64, dec_dim=64,
                 enc_layers=5, stu_layers=4, dec_layers=5, shortcut_layers=4,
                 inject_layers=4, stu_kernel_size=3, stu_inject_layers=4,
                 use_stu=True, multi_inputs=1, one_more_conv=0):
        super(G_stgan, self).__init__()
        self.Genc = Genc(dim=enc_dim, n_layers=enc_layers,
                         multi_inputs=multi_inputs)
        self.Gstu = Gstu(attr_dim, dim=stu_dim, enc_layers=enc_layers,
                         n_layers=stu_layers, inject_layers=stu_inject_layers,
                         kernel_size=stu_kernel_size)
        self.Gdec = Gdec(attr_dim, dim=dec_dim, n_layers=dec_layers,
                         shortcut_layers=shortcut_layers,
                         inject_layers=inject_layers,
                         one_more_conv=one_more_conv)

    def forward(self, x, _a):
        zs = self.Genc(x)
        zs_ = self.Gstu(zs, _a)
        z = self.Gdec(zs_, _a)
        return z


class D_stgan(nn.Module):
    def __init__(self, image_size, att_dim, dim=64, fc_dim=MAX_DIM,
                 n_layers=5):
        super(D_stgan, self).__init__()
        layers = []
        in_channels = 3
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            layers.append(conv_norm_active(in_channels, d, 4, 2, 1,
                                           norm_mode='in', act_mode='lrelu'))
            in_channels = d

        self.conv = nn.Sequential(*layers)
        feature_size = image_size // 2**n_layers
        d = min(dim * 2**i, MAX_DIM)
        self.logit_gan = nn.Sequential(
            nn.Linear(d * feature_size ** 2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, 1)
        )
        self.logit_att = nn.Sequential(
            nn.Linear(d * feature_size ** 2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, att_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        logit_gan = self.logit_gan(x)
        logit_att = self.logit_att(x)
        return logit_gan, logit_att


def gradient_penalty(f, real, fake=None):
    def _interpolate(a, b=None):
        if b is None:   # interpolation in DRAGAN
            beta = torch.rand(a.size())
            variance = torch.var(a)
            b = a + 0.5 * torch.sqrt(variance) * beta
        alpha = torch.rand(a.size())
        inter = a + alpha * (b - a)
        return inter

    x = _interpolate(real, fake)
    y = f(x)
    if isinstance(y, tuple):
        y = y[0]
    weight = torch.ones(y.size())
    grad = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]
    grad = grad.view(grad.size(0), -1)
    grad_l2norm = torch.sqrt(torch.sum(grad**2, dim=1))
    return torch.mean((grad_l2norm - 1)**2)
