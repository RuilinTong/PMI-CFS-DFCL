# -*-coding:utf8-*-

import torch
import os


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, activation='elu',
                 pool_kernel_size=(2, 2)):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding, stride)
        if activation == 'elu':
            self.act = torch.nn.ELU()
        elif activation == 'leaky':
            self.act = torch.nn.LeakyReLU()
        else:
            self.act = torch.nn.ReLU()
        self.pool = torch.nn.AvgPool2d(pool_kernel_size)  # use average pooling to improve stability.

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x)
        return x


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, up_factor, activation='elu'):
        super(DecoderBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channel, out_channel, (3, 3), padding=1)
        self.up_sample = torch.nn.Upsample(scale_factor=up_factor, mode='nearest')
        if activation == 'elu':
            self.act = torch.nn.ELU()
        elif activation == 'leaky':
            self.act = torch.nn.LeakyReLU()
        else:
            self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv(x))
        x = self.up_sample(x)
        return x


class VAE(torch.nn.Module):
    def __init__(self, blocks, in_shape, z_num=1, expand_dim=True, use_pool=False, z_dim=32):
        super(VAE, self).__init__()
        self.z_num = z_num
        self.z_dim = z_dim
        self.use_pool = use_pool
        # Encoder
        self.encoder_blocks = []
        if expand_dim:
            out_ch = 64
        else:
            out_ch = in_shape[0]
        for i in range(blocks):
            if i == 0:
                block = ConvBlock(
                    in_channels=in_shape[0], out_channels=out_ch, kernel_size=(3, 3), padding=1, stride=1)
            elif i == blocks - 1:
                block = ConvBlock(
                    in_channels=out_ch, out_channels=self.z_dim, kernel_size=(3, 3), padding=1, stride=1)
            else:
                block = ConvBlock(
                    in_channels=out_ch, out_channels=out_ch * 2 if expand_dim else out_ch,
                    kernel_size=(3, 3), padding=1, stride=1)
                if expand_dim:
                    out_ch = out_ch * 2
                else:
                    out_ch = in_shape[0]
            self.encoder_blocks.append(block)
        self.encoder = torch.nn.Sequential(*self.encoder_blocks)

        # Decoder
        self.decoder_blocks = []
        if expand_dim:
            out_ch = 64
        else:
            out_ch = in_shape[0]
        for i in range(blocks):
            if i == 0:
                block = DecoderBlock(in_channel=int(self.z_dim // 2), out_channel=out_ch, up_factor=2, activation='elu')
            elif i == blocks - 1:
                block = DecoderBlock(in_channel=out_ch, out_channel=out_ch, up_factor=2, activation='elu')
            else:
                block = DecoderBlock(
                    in_channel=out_ch,
                    out_channel=out_ch * 2 if expand_dim else out_ch,
                    up_factor=2,
                    activation='elu'
                )
                if expand_dim:
                    out_ch = out_ch * 2
                else:
                    out_ch = in_shape[0]
            self.decoder_blocks.append(block)
        self.decoder = torch.nn.Sequential(*self.decoder_blocks)
        self.final_decode_mean = torch.nn.Conv2d(out_ch, in_shape[0], (3, 3), padding=1)

        # loss function
        self.recon_loss = torch.nn.MSELoss()
        self.pool_layer = torch.nn.AdaptiveAvgPool2d(1)

    def encode(self, x):
        x = self.encoder(x)
        # output shape - batch_size x 16 x 8 x 8
        return x[:, :int(self.z_dim // 2), :, :], x[:, int(self.z_dim // 2):, :, :]

    def reparameterize(self, mu, logvar):
        if self.training:
            sample_z = []
            for i in range(self.z_num):
                # multiply log variance with 0.5, then in-place exponent yielding the standard deviation
                eps = torch.randn(size=mu.shape, dtype=torch.float32).to(mu.device)
                std = logvar.mul(0.5).exp_()
                sample_z.append(eps.mul(std).add_(mu))
            return sample_z
        else:
            return mu

    def decode(self, z):
        if self.training:
            recon_x = []
            for zi in z:
                out = self.decoder(zi)
                out = self.final_decode_mean(out)
                recon_x.append(out)
        else:
            recon_x = self.decoder(z)
            recon_x = self.final_decode_mean(recon_x)
        return recon_x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)  # sample multiple z
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = 0
        for recon_x_i in recon_x:
            if self.use_pool:
                recon_loss += self.recon_loss(self.pool_layer(recon_x_i)[:, :, 0, 0], self.pool_layer(x)[:, :, 0, 0])
            else:
                recon_loss += self.recon_loss(recon_x_i, x)
        recon_loss = recon_loss / len(recon_x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # kl_loss = -0.5 * torch.mean(1 + torch.mean(logvar, dim=0)
        #                             - torch.mean(mu, dim=0).pow(2)
        #                             - torch.mean(logvar.exp(), dim=0))
        return recon_loss, kl_loss


class SimpleVAE(torch.nn.Module):
    def __init__(self, blocks, in_shape, z_num=1, expand_dim=True):
        super(SimpleVAE, self).__init__()
        self.z_num = z_num
        # Encoder
        self.encoder_blocks = []
        if expand_dim:
            out_ch = 64
        else:
            out_ch = in_shape[0]
        for i in range(blocks):
            if i == 0:
                block = ConvBlock(
                    in_channels=in_shape[0],
                    out_channels=out_ch,
                    kernel_size=(3, 3),
                    padding=1,
                    stride=1
                )
            elif i == blocks - 1:
                block = ConvBlock(
                    in_channels=out_ch,
                    out_channels=32,
                    kernel_size=(3, 3),
                    padding=1,
                    stride=1)
            else:
                block = ConvBlock(
                    in_channels=out_ch,
                    out_channels=out_ch * 2 if expand_dim else out_ch,
                    kernel_size=(3, 3),
                    padding=1,
                    stride=1
                )
                if expand_dim:
                    out_ch = out_ch * 2
                else:
                    out_ch = in_shape[0]
            self.encoder_blocks.append(block)
        self.encoder = torch.nn.Sequential(*self.encoder_blocks)

        # Decoder
        self.decoder_blocks = []
        if expand_dim:
            out_ch = 64
        else:
            out_ch = in_shape[0]
        for i in range(blocks):
            if i == 0:
                block = DecoderBlock(in_channel=32, out_channel=out_ch, up_factor=2, activation='elu')
            elif i == blocks - 1:
                block = DecoderBlock(in_channel=out_ch, out_channel=out_ch, up_factor=2, activation='elu')
            else:
                block = DecoderBlock(
                    in_channel=out_ch,
                    out_channel=out_ch * 2 if expand_dim else out_ch,
                    up_factor=2,
                    activation='elu'
                )
                if expand_dim:
                    out_ch = out_ch * 2
                else:
                    out_ch = in_shape[0]
            self.decoder_blocks.append(block)
        self.decoder = torch.nn.Sequential(*self.decoder_blocks)
        self.final_decode_mean = torch.nn.Conv2d(out_ch, in_shape[0], (3, 3), padding=1)

        # loss function
        self.recon_loss = torch.nn.MSELoss()

    def encode(self, x):
        x = self.encoder(x)
        return x  # output shape - batch_size x 16 x 8 x 8

    def decode(self, z):
        recon_x = self.decoder(z)
        recon_x = self.final_decode_mean(recon_x)
        return recon_x

    def forward(self, x):
        rep = self.encode(x)
        recon_x = self.decode(rep)  # sample multiple z
        return rep, recon_x

    def loss_function(self, recon_x, x, rep):
        mu = torch.mean(rep, dim=0)
        var = torch.var(rep, dim=0, unbiased=True)
        recon_loss = self.recon_loss(recon_x, x)
        recon_loss = recon_loss / len(recon_x)
        kl_loss = -0.5 * torch.mean(1 + torch.log(var) - mu.pow(2) - var)
        return recon_loss, kl_loss


class SimpleMLPVAE(torch.nn.Module):
    def __init__(self, blocks, in_shape, z_num=1):
        super(SimpleMLPVAE, self).__init__()
        self.z_num = z_num
        # Encoder
        self.encoder_blocks = []
        out_ch = in_shape[0]
        for i in range(blocks):
            block = torch.nn.Linear(in_features=out_ch, out_features=out_ch, bias=True)
            self.encoder_blocks.append(block)
            act = torch.nn.LeakyReLU()
            self.encoder_blocks.append(act)
        self.encoder = torch.nn.Sequential(*self.encoder_blocks)

        # Decoder
        self.decoder_blocks = []
        out_ch = in_shape[0]
        for i in range(blocks):
            block = torch.nn.Linear(in_features=out_ch, out_features=out_ch, bias=True)
            self.decoder_blocks.append(block)
            act = torch.nn.LeakyReLU()
            self.decoder_blocks.append(act)
        self.decoder = torch.nn.Sequential(*self.decoder_blocks)

        # loss function
        self.recon_loss = torch.nn.MSELoss()

    def encode(self, x):
        x = self.encoder(x)
        return x  # output shape - batch_size x 16 x 8 x 8

    def decode(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def forward(self, x):
        rep = self.encode(x)
        recon_x = self.decode(rep)  # sample multiple z
        return rep, recon_x

    def loss_function(self, recon_x, x, rep):
        mu = torch.mean(rep, dim=0)
        var = torch.var(rep, dim=0, unbiased=True)
        recon_loss = self.recon_loss(recon_x, x)
        recon_loss = recon_loss / len(recon_x)
        kl_loss = -0.5 * torch.mean(1 + torch.log(var) - mu.pow(2) - var)
        return recon_loss, kl_loss


class SimpleCNNEncoder(torch.nn.Module):
    def __init__(self, in_shape, blocks):
        super().__init__()
        self.n_ch = in_shape[0]
        self.blocks = []
        for i in range(blocks):
            block = torch.nn.Conv2d(in_channels=self.n_ch, out_channels=self.n_ch, padding=1, kernel_size=3, bias=True)
            self.blocks.append(block)
            bn = torch.nn.BatchNorm2d(num_features=self.n_ch)
            self.blocks.append(bn)
            act = torch.nn.LeakyReLU()
            self.blocks.append(act)
        self.encoder = torch.nn.Sequential(*self.blocks)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        y = self.encoder(x)
        y = self.flatten(y)
        y = y / (torch.norm(y, p=2, dim=1, keepdim=True) + 1e-6)
        return y


class NegativeContrastiveLoss(torch.nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, x):  # x.shape = [N, dim]
        x_1 = torch.unsqueeze(x, dim=0)  # shape = [1, N, dim]
        x_2 = torch.unsqueeze(x, dim=1)  # shape = [N, 1, dim]
        cos = torch.sum(x_1 * x_2, dim=2) / self.tau  # shape = [N, N]
        exp_cos = torch.exp(cos)
        loss = torch.log(torch.mean(exp_cos, dim=1))  # diagonal elements are positive pairs
        loss = torch.mean(loss)
        return loss
