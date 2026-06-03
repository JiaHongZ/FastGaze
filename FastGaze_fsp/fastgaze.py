import torch
import torch.nn.functional as F
from torch import nn, Tensor

from positional_encodings import PositionEmbeddingSine2d

import torch.nn as nn
import torch.nn.init as init

class FEF(nn.Module):
    def __init__(self, hidden_dim, device):
        super(FEF, self).__init__()
        self.hidden_dim = hidden_dim
        # Define the layers
        self.generator_y_mu = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_x_mu = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_t_mu = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_y_logvar = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_x_logvar = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_t_logvar = nn.Linear(self.hidden_dim, 1).to(device)

        # Apply random initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights using Xavier uniform distribution for linear layers
        init.xavier_uniform_(self.generator_y_mu.weight)
        init.xavier_uniform_(self.generator_x_mu.weight)
        init.xavier_uniform_(self.generator_t_mu.weight)
        init.xavier_uniform_(self.generator_y_logvar.weight)
        init.xavier_uniform_(self.generator_x_logvar.weight)
        init.xavier_uniform_(self.generator_t_logvar.weight)
        
        # Initialize biases with zeros
        # init.zeros_(self.generator_y_mu.bias)
        # init.zeros_(self.generator_x_mu.bias)
        # init.zeros_(self.generator_t_mu.bias)
        # init.zeros_(self.generator_y_logvar.bias)
        # init.zeros_(self.generator_x_logvar.bias)
        # init.zeros_(self.generator_t_logvar.bias)

        value_x = 160 # Replace this with your desired fixed value
        value_y = 256 # Replace this with your desired fixed value

        init.constant_(self.generator_y_mu.bias, value_y)
        init.constant_(self.generator_x_mu.bias, value_x)
        init.zeros_(self.generator_t_mu.bias)
        init.zeros_(self.generator_y_logvar.bias)
        init.zeros_(self.generator_x_logvar.bias)
        init.zeros_(self.generator_t_logvar.bias)


    def forward(self, outs):
        y_mu, y_logvar, x_mu, x_logvar, t_mu, t_logvar = self.generator_y_mu(outs), self.generator_y_logvar(outs), self.generator_x_mu(outs), self.generator_x_logvar(outs), self.generator_t_mu(outs), self.generator_t_logvar(outs)
        return y_mu, y_logvar, x_mu, x_logvar, t_mu, t_logvar



class fastgaze(nn.Module):
    def __init__(self, transformer, spatial_dim, dropout=0.4, max_len=7, patch_size=16, device="cuda:0"):
        super(fastgaze, self).__init__()
        self.spatial_dim = spatial_dim
        self.vs = transformer.to(device)
        self.hidden_dim = transformer.d_model

        # Fixation embeddings
        self.querypos_embed = nn.Embedding(max_len, self.hidden_dim).to(device)  # Add one for class token

        # 2D patch positional encoding
        self.patchpos_embed = PositionEmbeddingSine2d(spatial_dim, hidden_dim=self.hidden_dim, normalize=True, device=device)

        # 2D pixel positional encoding for initial fixation
        self.queryfix_embed = PositionEmbeddingSine2d(
            (spatial_dim[0] * patch_size, spatial_dim[1] * patch_size),
            hidden_dim=self.hidden_dim,
            normalize=True,
            flatten=False,
            device=device,
        ).pos.to(device)

        # Classify fixation or PAD tokens
        self.token_predictor = nn.Linear(self.hidden_dim, 2).to(device)

        self.device = device
        self.max_len = max_len
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.LogSoftmax(dim=-1).to(device)

        # Projection for first fixation encoding
        self.fef = FEF(self.hidden_dim, device)
        self.task_predictor = nn.Linear(self.hidden_dim, 3).to(device)
        
        # Task type for dynamic Prompt generation
        self.task_embedding = nn.Embedding(3, transformer.lm_dmodel).to(device)  # 3 tasks: TP(0), TA(1), FV(2)

    # Reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src: Tensor, tgt: Tensor, task: Tensor, type_tensor):
        src = src.to(self.device)
        
        # 获取任务类型对应的 Prompt（通过任务类型索引从 task_embedding 中提取）
        task_prompts = self.task_embedding(type_tensor.squeeze().long().to(self.device)) # Shape: [b, hidden_dim]
        task = task.to(self.device) + task_prompts # 原本的加入了scanpath type的embedding
        # task = task.to(self.device) #warn: 修改后的
        # Initialize target input with zeros, now accommodating the class token
        tgt_input = torch.zeros(self.max_len, src.size(0), self.hidden_dim).to(self.device)
        # print(tgt_input.shape) # 7 32 512 -- 32是b
        # tgt_input[0, :, :] = self.firstfix_linear(self.queryfix_embed[tgt[:, 0], tgt[:,1], :])
        tgt_input[0, :, :] = self.queryfix_embed[tgt[:, 0], tgt[:,1], :]

        # Pass through the transformer
        outs = self.vs(
            src=src,
            tgt=tgt_input,
            tgt_key_padding_mask=None,
            task=task.to(self.device),
            querypos_embed=self.querypos_embed.weight.unsqueeze(1),
            patchpos_embed=self.patchpos_embed,
        )

        outs = self.dropout(outs) # torch.Size([8, 64, 256])
        y_mu, y_logvar, x_mu, x_logvar, t_mu, t_logvar = self.fef(outs)
        return self.softmax(self.task_predictor(outs.mean(0))), self.softmax(self.token_predictor(outs)), self.activation(self.reparameterize(y_mu, y_logvar)),self.activation(self.reparameterize(x_mu, x_logvar)), self.activation(self.reparameterize(t_mu, t_logvar))

