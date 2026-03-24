import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class RASFinderModel(nn.Module):

    def __init__(
        self,
        input_dim=23,
        summary_dim=10,
        num_classes=4,
        num_heads=2,
        num_layers=4,
        dim_model=238,
        dim_feedforward=299,
        use_checkpoint=True
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.fc1 = nn.Linear(input_dim, dim_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim_model + summary_dim, 33)
        self.fc3 = nn.Linear(33, num_classes)

    def forward(self, sitewise_feature, lengths, summary_feature):

        B, max_n_sites, _ = sitewise_feature.shape
        device = sitewise_feature.device

        x = self.fc1(sitewise_feature)

        padding_mask = torch.arange(max_n_sites, device=device)[None, :] >= lengths[:, None]

        if self.use_checkpoint and self.training:
            x = checkpoint.checkpoint(
                lambda y: self.transformer_encoder(y, src_key_padding_mask=padding_mask),
                x
            )
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        mask = (~padding_mask).unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        mean = summed / mask.sum(dim=1).clamp(min=1)

        x = torch.cat([mean, summary_feature], dim=-1)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class RASFinderV2(nn.Module):
    def __init__(
        self,
        input_dim,
        summary_dim,
        dim_model=128,
        num_heads=2,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1
    ):
        super().__init__()

        # ===== site encoder =====
        self.site_proj = nn.Linear(input_dim, dim_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ===== summary encoder =====
        self.summary_proj = nn.Sequential(
            nn.Linear(summary_dim, dim_model),
            nn.ReLU(),
            nn.LayerNorm(dim_model)
        )

        # ===== fusion =====
        self.fusion = nn.Sequential(
            nn.Linear(dim_model * 2, dim_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ===== dual heads =====
        self.head_G = nn.Linear(dim_model, 1)
        self.head_I = nn.Linear(dim_model, 1)

    def forward(self, site_feat, summary_feat):
        """
        site_feat: (B, n_sites, input_dim)
        summary_feat: (B, summary_dim)
        """

        x = self.site_proj(site_feat)

        x = self.transformer(x)

        x = x.mean(dim=1)

        s = self.summary_proj(summary_feat)

        h = torch.cat([x, s], dim=1)

        h = self.fusion(h)

        logit_G = self.head_G(h)
        logit_I = self.head_I(h)

        logits = torch.cat([logit_G, logit_I], dim=1)

        return logits

