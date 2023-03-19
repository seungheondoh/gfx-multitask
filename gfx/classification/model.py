import torch
import torchaudio
from torch import nn
from mtr.modules.backbone import Transformer

class MusicTaggingTransformer(nn.Module):
    def __init__(self,
                audio_representation,
                tokenizer, 
                spec_type,
                is_vq=False, 
                dropout=0.1, 
                attention_ndim=256,
                attention_nheads=8,
                attention_nlayers=4,
                attention_max_len=512,
                n_seq_cls= 50
        ):
        super(MusicTaggingTransformer, self).__init__()
        # Input preprocessing
        self.audio_representation = audio_representation
        self.spec_type = spec_type
        # Input embedding
        self.tokenizer = tokenizer
        self.is_vq = is_vq
        self.vq_modules = None
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, attention_max_len + 1, attention_ndim))
        self.cls_token = nn.Parameter(torch.randn(attention_ndim))
        # transformer
        self.transformer = Transformer(
            attention_ndim,
            attention_nlayers,
            attention_nheads,
            attention_ndim // attention_nheads,
            attention_ndim * 4,
            dropout,
        )
        self.to_latent = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        # projection for sequence classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(attention_ndim), nn.Linear(attention_ndim, n_seq_cls)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args: x (torch.Tensor): (batch, time)
        Returns: x (torch.Tensor): (batch, n_seq_cls)
        """
        # Input preprocessing
        if self.spec_type == "mel":
            spec = self.audio_representation.melspec(x)
            spec = spec.unsqueeze(1)
        elif self.spec_type == "stft":
            spec = None
        h_audio = self.tokenizer(spec) # B x L x D
        if self.is_vq:
            h_audio = self.vq_modules(h_audio)
        # Positional embedding with a [CLS] token
        cls_token = self.cls_token.repeat(h_audio.shape[0], 1, 1)
        h_audio = torch.cat((cls_token, h_audio), dim=1)
        h_audio += self.pos_embedding[:, : h_audio.size(1)]
        h_audio = self.dropout(h_audio)
        # transformer
        z_audio = self.transformer(h_audio)
        # projection for sequence classification
        e_audio = self.to_latent(z_audio[:, 0])
        output = self.mlp_head(e_audio)
        output = self.sigmoid(output)
        return output
        