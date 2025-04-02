from abc import abstractmethod

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin

from network.layout_diffusion.layout_encoder import LayoutTransformerEncoder
from network.layout_diffusion.modules import SiLU, conv_nd, linear, avg_pool_nd, zero_module, normalization, timestep_embedding


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, cond_kwargs=None):
        extra_output = None
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, ObjectAwareCrossAttention):
                x, extra_output = layer(x, cond_kwargs)
            else:
                x = layer(x)
        return x, extra_output


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class ObjectAwareCrossAttention(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            encoder_channels=None,
            return_attention_embeddings=False,
            ds=None,
            resolution=None,
            type=None,
            use_positional_embedding=True,
            use_key_padding_mask=False,
            channels_scale_for_positional_embedding=1.0,
            norm_first=False,
            norm_for_obj_embedding=False
    ):
        super().__init__()
        self.norm_for_obj_embedding=None
        self.norm_first = norm_first
        self.channels_scale_for_positional_embedding = channels_scale_for_positional_embedding
        self.use_key_padding_mask=use_key_padding_mask
        self.type = type
        self.ds = ds
        self.resolution = resolution.tolist()
        self.return_attention_embeddings = return_attention_embeddings

        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_positional_embedding = use_positional_embedding
        assert self.use_positional_embedding

        self.use_checkpoint = use_checkpoint

        self.qkv_projector = conv_nd(1, channels, 3 * channels, 1)
        self.norm_for_qkv = normalization(channels)

        if encoder_channels is not None:
            self.encoder_channels= encoder_channels
            self.layout_content_embedding_projector = conv_nd(1, encoder_channels, channels * 2, 1)
            self.layout_position_embedding_projector = conv_nd(1, encoder_channels, int(channels * self.channels_scale_for_positional_embedding), 1)
            if self.norm_first:
                if norm_for_obj_embedding:
                    self.norm_for_obj_embedding = normalization(encoder_channels)
                self.norm_for_obj_class_embedding = normalization(encoder_channels)
                self.norm_for_layout_positional_embedding = normalization(encoder_channels)
                self.norm_for_image_patch_positional_embedding = normalization(encoder_channels)
            else:
                self.norm_for_obj_class_embedding = normalization(encoder_channels)
                self.norm_for_layout_positional_embedding = normalization(int(channels * self.channels_scale_for_positional_embedding))
                self.norm_for_image_patch_positional_embedding = normalization(int(channels * self.channels_scale_for_positional_embedding))

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond_kwargs):
        '''
        :param x: (N, C, H, W)
        :param cond_kwargs['xf_out']: (N, C, L2)
        :return:
            extra_output: N x L2 x 3 x ds x ds
        '''
        extra_output = None
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # N x C x (HxW)

        # Q_i, K_i, V_i embeddings from image features I
        qkv = self.qkv_projector(self.norm_for_qkv(x))  # N x 3C x L1 (L1=H*W)
        bs, C, L1, L2 = qkv.shape[0], self.channels, qkv.shape[2], cond_kwargs['obj_bbox_embedding'].shape[-1]
        q_image_patch_content_embedding, k_image_patch_content_embedding, v_image_patch_content_embedding = qkv.split(C, dim=1)  # 3 x (N , C, L1)
        q_image_patch_content_embedding = q_image_patch_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L1)  # (N // num_heads, C // num_heads, L1)
        k_image_patch_content_embedding = k_image_patch_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L1)  # (N // num_heads, C // num_heads, L1)
        v_image_patch_content_embedding = v_image_patch_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L1)  # (N // num_heads, C // num_heads, L1)

        # P_i positional embedding for image patch b_i
        if self.norm_first:
            image_patch_positional_embedding = self.norm_for_image_patch_positional_embedding(
                cond_kwargs['image_patch_bbox_embedding_for_resolution{}'.format(self.resolution)]
                )  # N, encoder_channels, L1
            image_patch_positional_embedding = self.layout_position_embedding_projector(image_patch_positional_embedding)  # N x C * channels_scale_for_positional_embedding x L1, (L1=H*W)
        else:
            image_patch_positional_embedding = self.layout_position_embedding_projector(
                cond_kwargs['image_patch_bbox_embedding_for_resolution{}'.format(self.resolution)]
            )  # N x C * channels_scale_for_positional_embedding x L1, (L1=H*W)
            image_patch_positional_embedding = self.norm_for_image_patch_positional_embedding(image_patch_positional_embedding)  # (N, C * channels_scale_for_positional_embedding, L1)
        image_patch_positional_embedding = image_patch_positional_embedding.reshape(
            bs * self.num_heads, 
            int(C * self.channels_scale_for_positional_embedding) // self.num_heads, 
            L1
        )  # (N * num_heads, C * channels_scale_for_positional_embedding // num_heads, L1)
        
        # [Q_i, P_i], [K_i, P_i], [V_i] embeddings
        q_image_patch = torch.cat([q_image_patch_content_embedding, image_patch_positional_embedding], dim=1)  # (N // num_heads, (1+channels_scale_for_positional_embedding) * C // num_heads, L1)
        k_image_patch = torch.cat([k_image_patch_content_embedding, image_patch_positional_embedding], dim=1)  # (N // num_heads, (1+channels_scale_for_positional_embedding) * C // num_heads, L1)
        v_image_patch = v_image_patch_content_embedding  # (N // num_heads, C // num_heads, L1)

        # P_l positional embedding for bbox embeddings B_l
        if self.norm_first:
            layout_positional_embedding = self.norm_for_layout_positional_embedding(cond_kwargs['obj_bbox_embedding'])  # (N, encoder_channels, L2)
            layout_positional_embedding = self.layout_position_embedding_projector(layout_positional_embedding)  # N x C*channels_scale_for_positional_embedding x L2
        else:
            layout_positional_embedding = self.layout_position_embedding_projector(cond_kwargs['obj_bbox_embedding'])  # N x C*channels_scale_for_positional_embedding x L2
            layout_positional_embedding = self.norm_for_layout_positional_embedding(layout_positional_embedding)  # (N, C * channels_scale_for_positional_embedding, L2)
        layout_positional_embedding = layout_positional_embedding.reshape(bs * self.num_heads, int(C * self.channels_scale_for_positional_embedding) // self.num_heads, L2)  # (N // num_heads, channels_scale_for_positional_embedding * C // num_heads, L2)

        # K_l, V_l embeddings from layout embeddings L' and class embeddings C_l
        if self.norm_for_obj_embedding is not None:
            layout_content_embedding = (self.norm_for_obj_embedding(cond_kwargs['xf_out']) + self.norm_for_obj_class_embedding(cond_kwargs['obj_class_embedding'])) / 2
        else:
            layout_content_embedding = (cond_kwargs['xf_out'] + self.norm_for_obj_class_embedding(cond_kwargs['obj_class_embedding'])) / 2
        k_layout_content_embedding, v_layout_content_embedding = self.layout_content_embedding_projector(layout_content_embedding).split(C, dim=1)  # 2 x (N x C x L2)
        k_layout_content_embedding = k_layout_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L2)  # (N // num_heads, C // num_heads, L2)
        v_layout_content_embedding = v_layout_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L2)  # (N // num_heads, C // num_heads, L2)

        # [K_l, P_l], [V_l] embedding for layout
        k_layout = torch.cat([k_layout_content_embedding, layout_positional_embedding], dim=1)  # (N // num_heads, (1+channels_scale_for_positional_embedding) * C // num_heads, L2)
        v_layout = v_layout_content_embedding  # (N // num_heads, C // num_heads, L2)

        # [K_i, P_i, K_l, P_l], [V_i, V_l] embeddings for cross attention
        k_mix = torch.cat([k_image_patch, k_layout], dim=2)  # (N // num_heads, (1+channels_scale_for_positional_embedding) * C // num_heads, L1+L2)
        v_mix = torch.cat([v_image_patch, v_layout], dim=2)  # (N // num_heads, 1 * C // num_heads, L1+L2)

        ################################
        # CROSS ATTENTION MODULE START #
        ################################
        if self.use_key_padding_mask:
            key_padding_mask = torch.cat(
                [
                    torch.zeros((bs, L1), device=cond_kwargs['key_padding_mask'].device).bool(),  # (N, L1)
                    cond_kwargs['key_padding_mask']  # (N, L2)
                ],
                dim=1
            )  # (N, L1+L2)
            print(cond_kwargs['key_padding_mask'])

        scale = 1 / math.sqrt(math.sqrt(int((1+self.channels_scale_for_positional_embedding) * C) // self.num_heads))
        # More stable with f16 than dividing afterwards, (N x num_heads, L1, L1+L2)
        attn_output_weights = torch.einsum("bct,bcs->bts", q_image_patch * scale, k_mix * scale)

        attn_output_weights = attn_output_weights.view(bs, self.num_heads, L1, L1 + L2)

        if self.use_key_padding_mask:
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (N, 1, 1, L1+L2)
                float('-inf'),
            )
        attn_output_weights = attn_output_weights.view(bs * self.num_heads, L1, L1 + L2)

        attn_output_weights = torch.softmax(attn_output_weights.float(), dim=-1).type(attn_output_weights.dtype)  # (N x num_heads, L1, L1+L2)

        attn_output = torch.einsum("bts,bcs->bct", attn_output_weights, v_mix)  # (N x num_heads, C // num_heads, L1)
        attn_output = attn_output.reshape(bs, C, L1)  # (N, C, L1)

        h = self.proj_out(attn_output)
        ##############################
        # CROSS ATTENTION MODULE END #
        ##############################

        # Image features (with global information as positional embedding) + cross attention output
        output = (x + h).reshape(b, c, *spatial)

        if self.return_attention_embeddings:
            assert cond_kwargs is not None
            if extra_output is None:
                extra_output = {}
            extra_output.update({
                'type': self.type,
                'ds': self.ds,
                'resolution': self.resolution,
                'num_heads': self.num_heads,
                'num_channels': self.channels,
                'image_query_embeddings': image_patch_positional_embedding.detach().view(bs, -1, L1),  # N x C x L1
                # 'image_query_embeddings': qkv[:, :self.channels, :].detach(),  # N x C x L1
            })
            if cond_kwargs is not None:
                extra_output.update({
                    'layout_key_embeddings': layout_positional_embedding.detach().view(bs, -1, L2)  # N x C x L2

                    # 'layout_key_embeddings': kv_for_encoder_out[:, : self.channels, :].detach()  # N x C x L2
                })

        return output, extra_output


class LayoutDiffusionUNetModel(ModelMixin, nn.Module):
    """
    A UNetModel that conditions on layout with an encoding transformer.
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_ds: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.

    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param {
        layout_length: number of layout objects to expect.
        hidden_dim: width of the transformer.
        num_layers: depth of the transformer.
        num_heads: heads in the transformer.
        xf_final_ln: use a LayerNorm after the output layer.
        num_classes_for_layout_object: num of classes for layout object.
        mask_size_for_layout_object: mask size for layout object image.
    }

    """

    # Small configuration for layout diffusion unet
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            image_size=256,
            model_channels=128,
            num_res_blocks=2,
            attention_ds=[32, 16, 8],
            encoder_channels=128,
            dropout=0,
            channel_mult=(1, 1, 2, 2, 4, 4),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            num_heads=-1,
            num_head_channels=32,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_positional_embedding_for_attention=True,
            num_attention_blocks=1,
            use_key_padding_mask=False,
            channels_scale_for_positional_embedding=1.0,
            norm_first=False,
            norm_for_obj_embedding=False,
            layout_encoder=None
    ):
        super().__init__()
        self.norm_for_obj_embedding = norm_for_obj_embedding
        self.channels_scale_for_positional_embedding = channels_scale_for_positional_embedding
        self.norm_first = norm_first
        self.use_key_padding_mask=use_key_padding_mask
        self.num_attention_blocks = num_attention_blocks

        self.image_size = image_size
        self.use_positional_embedding_for_attention = use_positional_embedding_for_attention

        # Small configuration for layout transformer
        self.layout_encoder = LayoutTransformerEncoder(**layout_encoder)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_ds = attention_ds
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_ds:
                    # print('encoder attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
                    for _ in range(self.num_attention_blocks):
                        layers.append(
                            ObjectAwareCrossAttention(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                encoder_channels=encoder_channels,
                                ds=ds,
                                resolution=self.image_size // ds,
                                type='input',
                                use_positional_embedding=self.use_positional_embedding_for_attention,
                                use_key_padding_mask=self.use_key_padding_mask,
                                channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                                norm_first=self.norm_first,
                                norm_for_obj_embedding=self.norm_for_obj_embedding
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # print('middle attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ObjectAwareCrossAttention(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
                ds=ds,
                resolution=self.image_size // ds,
                type='middle',
                use_positional_embedding=self.use_positional_embedding_for_attention,
                use_key_padding_mask=self.use_key_padding_mask,
                channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                norm_first=self.norm_first,
                norm_for_obj_embedding=self.norm_for_obj_embedding
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_ds:
                    # print('decoder attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
                    for _ in range(self.num_attention_blocks):
                        layers.append(
                            ObjectAwareCrossAttention(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                encoder_channels=encoder_channels,
                                ds=ds,
                                resolution=self.image_size // ds,
                                type='output',
                                use_positional_embedding=self.use_positional_embedding_for_attention,
                                use_key_padding_mask=self.use_key_padding_mask,
                                channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                                norm_first=self.norm_first,
                                norm_for_obj_embedding=self.norm_for_obj_embedding
                            )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, **kwargs):
        hs, extra_outputs = [], []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Layout fusion module
        layout_outputs = self.layout_encoder(
            obj_class=obj_class,
            obj_bbox=obj_bbox,
            obj_mask=obj_mask,
            is_valid_obj=is_valid_obj
        )

        # Global condition (layout + timestep)
        emb = emb + layout_outputs["xf_proj"].to(emb)

        # UNet + Object-aware cross-attention
        h = x
        for module in self.input_blocks:
            h, extra_output = module(h, emb, layout_outputs)
            if extra_output is not None:
                extra_outputs.append(extra_output)
            hs.append(h)
        h, extra_output = self.middle_block(h, emb, layout_outputs)
        if extra_output is not None:
            extra_outputs.append(extra_output)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h, extra_output = module(h, emb, layout_outputs)
            if extra_output is not None:
                extra_outputs.append(extra_output)
        h = h.type(x.dtype)
        h = self.out(h)

        return [h, extra_outputs]


if __name__ == '__main__':
    bs = 4
    image_size = 256
    
    if image_size == 128:
        model = LayoutDiffusionUNetModel(image_size=image_size, channel_mult=[1, 1, 2, 3, 4], dropout=0.1)
    else:
        model = LayoutDiffusionUNetModel()
    print(model)
    
    # count number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters: %d' % num_params)

    model.cuda()
    
    x = torch.randn(bs, 3, image_size, image_size).cuda()
    timesteps = torch.randn(bs).cuda()
    obj_class = torch.ones(bs, 10).cuda()
    obj_bbox = torch.zeros([bs, 10, 4]).cuda()
    
    output, _ = model(x, timesteps, obj_class, obj_bbox)
    print(output[0].shape)
    