from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


# ResNet-50 backbone
class ResNetCOCO(nn.Module):
    def __init__(self, device="cuda:0"):
        super(ResNetCOCO, self).__init__()
        self.resnet = maskrcnn_resnet50_fpn(pretrained=True).backbone.body.to(device)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        bs, ch, _, _ = x.size()
        x = x.view(bs, ch, -1).permute(0, 2, 1)
        # print(x.shape)
        return x


from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn


class MobileNetCOCO(nn.Module):
    def __init__(self, device="cuda:0"):
        super(MobileNetCOCO, self).__init__()
        # 使用 Faster R-CNN with MobileNetV3 Large FPN backbone
        self.mobilenet = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True).backbone.body.to(device)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)

        # FasterRCNN backbone with FPN returns an OrderedDict of feature maps
        feature_dict = self.mobilenet(x)  # feature_dict 是一个 OrderedDict
        # 获取最后一层的卷积特征图 (通常是最后一个键 '2')
        # 注意：你可以用 list() 方法获取 OrderedDict 的所有键
        last_feature_map = feature_dict[list(feature_dict.keys())[-1]]  # 获取最后一层特征图
        # print(last_feature_map.shape)
        # 输出维度调整
        bs, ch, _, _ = last_feature_map.size()
        last_feature_map = last_feature_map.view(bs, ch, -1).permute(0, 2, 1)

        # print(last_feature_map.shape)
        return last_feature_map


class VS(nn.Module):

    def __init__(self, d_model=512, img_hidden_dim=2048, lm_dmodel=768, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=512, encoder_dropout=0.1, decoder_dropout=0.2,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, device="cuda:0", img_ior=False, img_mask=False, sc_mask=False,
                 txt_mask=False, sc_ior=False, txt_ior=False, max_len=7):
        super().__init__()
        self.device = device

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                encoder_dropout, activation, normalize_before, img_ior=img_ior,
                                                img_mask=img_mask).to(device)
        encoder_norm = nn.LayerNorm(d_model)
        encoder = ImgEncoder(encoder_layer, num_encoder_layers, encoder_norm).to(device)
        input_proj = nn.Linear(img_hidden_dim, d_model).to(device)
        img_transform = nn.Linear(d_model, d_model).to(device)
        text_transform = nn.Linear(lm_dmodel, d_model).to(device)
        self.lip = LIP(encoder, input_proj, img_transform, text_transform, encoder_dropout,
                                                 device).to(device)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                decoder_dropout, activation, normalize_before, sc_mask=sc_mask,
                                                txt_mask=txt_mask, sc_ior=sc_ior, txt_ior=txt_ior, max_len=max_len).to(
            device)
        decoder_norm = nn.LayerNorm(d_model)
        decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                     return_intermediate=return_intermediate_dec).to(device)

        dropout = nn.Dropout(decoder_dropout)
        self.sc = SC(d_model, activation, decoder, dropout, device)

        self.d_model = d_model
        self.lm_dmodel = lm_dmodel
        self.nhead = nhead

    def forward(self, src: Tensor, tgt: Tensor, task: Tensor, src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None, patchpos_embed: Optional[Tensor] = None):
        # print(src.shape, tgt.shape) # [32, 640, 2048] [7, 32, 512]
        memory, task_emb = self.lip(src, mask=src_mask, task=task, src_key_padding_mask=src_key_padding_mask,
                                        patchpos_embed=patchpos_embed)
        # print(memory.shape) # [640, 32, 512]
        # print(task_emb.shape) # [32, 512]
        output = self.sc(tgt, memory, task_emb=task_emb, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              querypos_embed=querypos_embed, patchpos_embed=patchpos_embed)
        return output


class LIP(nn.Module):

    def __init__(self, encoder, input_proj, img_transform, text_transform, dropout, device):
        super().__init__()
        self.device = device
        self.encoder = encoder.to(device)
        self.input_proj = input_proj.to(device)
        self.img_transform = img_transform.to(device)
        self.text_transform = text_transform.to(device)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters(self.encoder)
        self._reset_parameters(self.input_proj)
        self._reset_parameters(self.img_transform)
        self._reset_parameters(self.text_transform)

    def _reset_parameters(self, mod):
        for p in mod.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, task,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        src_proj = self.input_proj(src).permute(1, 0, 2)  # input projection from 2048 -> d
        # print(src.shape) # [32, 640, 2048]
        # print(src_proj.shape) # [640, 32, 512]
        output = self.encoder(src_proj, mask=mask, src_key_padding_mask=src_key_padding_mask,
                              patchpos_embed=patchpos_embed)  # transformer encoder
        # print(output.shape) # [640, 32, 512])
        memory = self.img_transform(output)  # project image features to multimodal space
        # print(memory.shape) # [640, 32, 512]
        task_emb = self.text_transform(task)  # project task features to multimodal space
        # print(task_emb.shape) # [32, 512]
        return memory, task_emb


class SC(nn.Module):
    def __init__(self, d_model, activation, decoder, dropout, device):
        super().__init__()
        self.device = device
        self.activation = _get_activation_fn(activation)
        self.decoder = decoder.to(device)
        self.dropout = dropout
        self.lineartgt = nn.Linear(d_model, d_model)

        self.linear = nn.Linear(d_model * 2, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos(tensor)

    def forward(self, tgt, memory, task_emb,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        # vision-semantic joint embedding
        memory_task = self.dropout(self.activation(
            self.linear(torch.cat([memory, task_emb.unsqueeze(0).repeat(memory.size(0), 1, 1)], dim=-1))))
        tgt = self.dropout(self.activation(
            self.lineartgt(tgt)))
        # decoder
        output = self.decoder(tgt, memory_task, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              querypos_embed=querypos_embed, patchpos_embed=patchpos_embed)
        return output


class ImgEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        output = src
        # print(output.size()) # [640, 32, 512]
        for layer in self.layers:
            output = layer(output, pos=patchpos_embed, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        # print(output.shape) # [640, 32, 512]

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for idx, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           querypos_embed=querypos_embed,
                           patchpos_embed=patchpos_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, img_ior=False, img_mask=False, max_len=7):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.img_ior = img_ior
        self.img_mask = img_mask
        self.max_len = max_len
        if img_ior:
            self.ior = nn.Parameter(torch.ones(640))  # 640维可学习向量

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos(tensor)

    def generate_ior(self, seq_len):
        # 截取或扩展 raw_influence_vector 以匹配 seq_len
        if seq_len <= 640:
            influence_vector_resized = self.ior[:seq_len]  # 如果序列较短，截取前 seq_len 个元素
        else:
            # 如果序列较长，重复 raw_influence_vector 以适配
            influence_vector_resized = self.ior.unsqueeze(0).repeat(seq_len // 640 + 1, 1).flatten()[:seq_len]

        # 生成 influence_mask 通过向量的外积 (outer product)
        influence_mask = torch.tril(torch.outer(influence_vector_resized, influence_vector_resized), diagonal=-1)
        influence_mask = -F.softplus(influence_mask)  # 确保为负值
        return influence_mask

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     patchpos_embed: Optional[Tensor] = None):

        q = k = self.with_pos_embed(src, patchpos_embed)

        # print(src_mask)
        # print(src.shape)
        if self.img_mask:
            seq_len = src.size(0)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
            src_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf')).masked_fill(causal_mask == 0,
                                                                                            float(0.0)).to(src.device)
            # print('img_mask')

        if self.img_ior:
            seq_len = src.size(0)
            ior = self.generate_ior(seq_len)
            src_mask = src_mask + ior
            # print('img_ior')
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)

        if self.img_mask:
            seq_len = src.size(0)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
            src_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf')).masked_fill(causal_mask == 0,
                                                                                            float(0.0)).to(src.device)
        if self.img_ior:
            seq_len = tgt.size(0)
            ior = self.generate_ior(seq_len)
            src_mask = src_mask + ior

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, sc_mask=False, txt_mask=False, sc_ior=False, txt_ior=False,
                 max_len=7):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.att_linear = nn.Linear(d_model, d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sc_mask = sc_mask
        self.txt_mask = txt_mask
        self.sc_ior = sc_ior
        self.txt_ior = txt_ior
        self.max_len = max_len
        self.nhead = nhead
        if sc_ior:
            self.sior = nn.Linear(d_model, max_len)  # 640维可学习向量
            
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos + tensor

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     querypos_embed: Optional[Tensor] = None,
                     patchpos_embed: Optional[Tensor] = None):
        # tgt = self.att_linear(tgt)
        # print(self.with_pos_embed(tgt, querypos_embed).shape, patchpos_embed(memory).shape, memory.shape) # torch.Size([7, 64, 512]) torch.Size([640, 64, 512]) torch.Size([640, 64, 512])
        # 多模态融合
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, querypos_embed),
                                   key=patchpos_embed(memory),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # # 因果mask和ior
        q = k = v = self.with_pos_embed(tgt, querypos_embed)
        if self.sc_mask:
            seq_len = tgt.size(0)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
            tgt_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf')).masked_fill(causal_mask == 0,
                                                                                            float(1)).to(tgt.device)
            # print('sc_mask', tgt_mask)

        if self.sc_ior:
            outer = self.sior(tgt.permute(1,0,2))
            tgt_mask = -F.softplus(outer)
            tgt_mask = torch.triu(tgt_mask, diagonal=1)
            b,t,t = tgt_mask.shape
            tgt_mask = tgt_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(b * self.nhead, t, t)

        tgt3 = self.multihead_attn2(q, k, value=v, attn_mask=tgt_mask,
                                    key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout3(tgt3)
        tgt = self.norm3(tgt)

        tgt3 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt3)
        tgt = self.norm4(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):


        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 querypos_embed=querypos_embed,
                                 patchpos_embed=patchpos_embed)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


