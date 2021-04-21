
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn


def _init_linear(layer, mean=0, std=0.01, bias=0):
    layer.weight.data = nn.init.normal_(layer.weight.data, mean=mean, std=std) 
    if layer.bias is not None: 
        layer.bias.data = nn.init.constant_(layer.bias.data, bias)


class AttentionHead(nn.Module):
    def __init__(self, in_features, out_features, max_seq_len, dropout=0):
        """Applies query, key, value matrix transformations to input features.
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            max_seq_len (int): Max sequence length for relative position embedding
            dropout (float): Dropout probability
        """
        
        super().__init__()
        
        self.n_out = out_features
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
        
        # Query, Key, and Value layers
        self.query = nn.Linear(in_features, out_features, bias=False)
        self.key = nn.Linear(in_features, out_features, bias=False)
        self.value = nn.Linear(in_features, out_features, bias=False)

        # Relative postion artifacts
        self.rel_pos = nn.parameter.Parameter(
            data=torch.cdist(
                torch.arange(max_seq_len)[:,None].to(torch.float), 
                torch.arange(max_seq_len)[:,None].to(torch.float), 
                p=1
            ).to(torch.long),
            requires_grad=False,
        )
        self.rel_pos_emb = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=1)

        # Softmax
        self.softmax = nn.Softmax(dim=-1)

        # Boom feed-forward layer
        self.ffn = nn.Sequential(
            nn.BatchNorm1d(num_features=out_features),
            nn.Linear(out_features, out_features*2),
            nn.ReLU(),
            nn.Linear(out_features*2, out_features),
        )
        
        self._init_wts()
        
    def _init_wts(self):
        for m in [self.query, self.key, self.value]:
            m.weight.data = nn.init.normal_(m.weight.data, mean=0, std=(1/m.in_features)**0.5)

        self.rel_pos_emb.weight.data = nn.init.normal_(self.rel_pos_emb.weight.data, mean=0, std=0.01)

    def _encode_rel_pos(self, seq_len):
        return self.rel_pos_emb(self.rel_pos[:seq_len,:seq_len]).permute(2,0,1)

    def get_attention_weights(self, x):
        # Reshape x for query and key transforms
        batch_size, seq_len, n_feats = x.shape
        x = x.reshape(-1, n_feats)
        
        # Calculate query/key vectors
        queries = self.query(x).view(batch_size, seq_len, self.n_out)
        keys = self.key(x).view(batch_size, seq_len, self.n_out)
        
        # Calculate attention scores 
        scores = torch.bmm(queries, keys.transpose(1, 2).contiguous())/self.n_out**0.5

        # Add relative position embedding
        scores = scores + self._encode_rel_pos(seq_len)
        
        # Calculate softmax of scores 
        weights = self.softmax(scores.view(-1, seq_len)).view(batch_size, seq_len, seq_len)
        
        return weights
    
    def forward(self, x):
        """
        x : (batch_size, seq_len, n_feats)
        """
        batch_size, seq_len, n_feats = x.shape
        x = self.dropout(x)

        # Calculate attention weights
        weights = self.get_attention_weights(x)
        
        # Calculate value vectors 
        values = self.value(x.reshape(-1, n_feats))
        values = values.reshape(batch_size, seq_len, self.n_out) 
         
        # Avg weighted avg values based on attention weights 
        values = torch.bmm(weights, values)

        # Feed-forward
        out = self.ffn(values.reshape(-1, self.n_out))
        out = out.reshape(batch_size, seq_len, self.n_out) 

        return out


class EncoderBlock(nn.Module):
    def __init__(
        self, 
        in_features,
        out_features,
        max_seq_len,
        rnn_kernel_size,
        dropout=0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.max_seq_len = max_seq_len

        self.attention_head = AttentionHead(in_features, in_features, max_seq_len, dropout)

        self.rnn_kernel_size = rnn_kernel_size
        self.rnn = nn.GRU(
            input_size=in_features, 
            hidden_size=out_features//2, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
    
    def forward(self, x):
        """
        Args:
            x (tensor): (batch_size, seq_len, in_features)

        Returns
            x (tensor): (bath_size, seq_len, in_features)
        """

        # Attention mixing + skip connection
        x = x + self.attention_head(x)
    
        # RNN pass
        batch_size, seq_len = x.shape[:2]
        x, _ = self.rnn(x.reshape(-1, self.rnn_kernel_size, self.in_features))
        x = x.reshape(batch_size, seq_len, self.out_features)

        return x


class PriceSeriesFeaturizer(nn.Module):
    def __init__(
        self, 
        n_features,
        historical_seq_len,
        future_seq_len,
        n_dist_targets,
        dropout=0):

        super().__init__()

        self.historical_seq_len = historical_seq_len
        self.future_seq_len = future_seq_len

        self.n_emb_feats = 8
        self.n_LCH_feats = 3
        self.pos_emb_dim = 8
        self.n_features = n_features

        self.n_dist_targets = n_dist_targets

        # Input embedder
        self.embeddings = nn.ModuleDict({
            'month': nn.Embedding(num_embeddings=12, embedding_dim=2),
            'dow': nn.Embedding(num_embeddings=5, embedding_dim=2),
            'trading_day': nn.Embedding(num_embeddings=23, embedding_dim=1),
            'trading_days_left': nn.Embedding(num_embeddings=23, embedding_dim=1),
            'earnings_marker': nn.Embedding(num_embeddings=5, embedding_dim=2),
            'future_pos_enc': nn.Embedding(num_embeddings=future_seq_len, embedding_dim=self.pos_emb_dim),
        })
        
        # Historical sequence encoder blocks
        self.hist_encoder_blocks = nn.ModuleList([
            EncoderBlock(
                in_features=self.n_emb_feats + self.n_LCH_feats,
                out_features=n_features//2,
                max_seq_len=historical_seq_len,
                rnn_kernel_size=8,
                dropout=dropout
            ),
            EncoderBlock(
                in_features=n_features//2,
                out_features=n_features,
                max_seq_len=historical_seq_len,
                rnn_kernel_size=16,
                dropout=dropout
            ),
        ])

        # Historical feature averager
        self.h_feat_averager = nn.parameter.Parameter(
            data=torch.randn(future_seq_len, historical_seq_len)*0.01,
            requires_grad=True
        )
        self.softmax = nn.Softmax(dim=-1)

        # Future position indxes
        self.pos_idxs = nn.parameter.Parameter(
            data=torch.arange(future_seq_len).unsqueeze(0),
            requires_grad=False
        )

        # Feature sequence encoder
        self.future_encoder = nn.GRU(
            input_size=self.n_emb_feats + self.pos_emb_dim + n_features, 
            hidden_size=n_features//2, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True,
        )

        # Low, Close, High mapper
        self.LCH_mapper = nn.Linear(in_features=n_features, out_features=self.n_LCH_feats)
        
        # Return distribution classifier
        self.dists_forecaster = nn.Sequential(
            nn.Linear(n_features, n_dist_targets),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_wts()

    def _init_wts(self):
        for emb in self.embeddings.values():
            emb.weight.data = nn.init.normal_(emb.weight.data, mean=0, std=0.01)

    def _emb_seq(self, x):
        """
        Args:
            x (tensor): Embedding sequence (batch_size, seq_len, 5)
        """
        embs = torch.cat([
            self.embeddings[k](x[:,:,i])
            for i, k 
            in enumerate(['month', 'dow', 'trading_day', 'trading_days_left', 'earnings_marker'])
        ], dim=-1)
        
        return embs

    def _encode_position(self, batch_size):
        """
        Args:
            batch_size (int)
        """

        pos_idxs = torch.repeat_interleave(self.pos_idxs, repeats=batch_size, dim=0)
        return self.embeddings['future_pos_enc'](pos_idxs)

    def _avg_wts(self):
        return self.softmax(self.h_feat_averager)
        
    def encode(self, x_emb, x_LCH):

        # Embed inputs and concat to LCH inputs
        x = torch.cat((self._emb_seq(x_emb), x_LCH), dim=-1)

        # Encode
        for block in self.hist_encoder_blocks:
            x = block(x)

        return x

    def encode_future_path(self, h_emb, h_LCH, f_emb):

        # Historical features
        h_feat = self.encode(h_emb, h_LCH)
        
        # Historical feature averages
        h_avg = torch.tensordot(
            self._avg_wts(), 
            h_feat, 
            dims=([1],[1])
        ).permute(1,0,2)

        # Cat future features
        batch_size = f_emb.shape[0]
        f_emb = self._emb_seq(f_emb)
        f_pos_enc = self._encode_position(batch_size)
        x = torch.cat([f_emb, f_pos_enc, h_avg], dim=-1)

        # Encode
        x, _ = self.future_encoder(x)

        return x

    def return_distribution_probas(self, f_feat):
        batch_size = f_feat.shape[0]
        
        # Reshape
        f_feat = f_feat.reshape(-1, self.n_features)

        # Probability predictions
        f_ret_probas = self.dists_forecaster(f_feat).reshape(batch_size, self.future_seq_len, self.n_dist_targets)

        return f_ret_probas

    def forward(self, h_emb, h_LCH, f_emb):
        
        batch_size = h_emb.shape[0]
        
        # Encode historical features
        h_feat = self.encode(h_emb, h_LCH)
        
        # Historical feature averages
        h_avg = torch.tensordot(
            self._avg_wts(), 
            h_feat, 
            dims=([1],[1])
        ).permute(1,0,2)

        # Encode future features
        f_emb = self._emb_seq(f_emb)
        f_pos_enc = self._encode_position(batch_size)
        f_feat = torch.cat([f_emb, f_pos_enc, h_avg], dim=-1)
        f_feat, _ = self.future_encoder(f_feat)

        # Reshape
        h_feat = h_feat.reshape(-1, self.n_features)
        f_feat = f_feat.reshape(-1, self.n_features)

        # LCH  predictions
        h_LCH = self.LCH_mapper(h_feat).reshape(batch_size, self.historical_seq_len, self.n_LCH_feats)
        f_LCH = self.LCH_mapper(f_feat).reshape(batch_size, self.future_seq_len, self.n_LCH_feats)

        # Probability predictions
        f_ret_probas = self.dists_forecaster(f_feat).reshape(batch_size, self.future_seq_len, self.n_dist_targets)

        return h_LCH, f_LCH, f_ret_probas


class AdaptiveInstanceNormalization(nn.Module):
    def __init__(self, seq_len, n_features):
        super().__init__()

        self.scale_mapper = nn.Sequential(
            nn.Linear(n_features, n_features//2),
            nn.ReLU(),
            nn.Linear(n_features//2, seq_len)
        )

        self.bias_mapper = nn.Sequential(
            nn.Linear(n_features, n_features//2),
            nn.ReLU(),
            nn.Linear(n_features//2, seq_len)
        )

    def forward(self, x, noise):
        """
        context (tensor): (batch_size, seq_len, n_features)
        noise (tensor): (batch_size, n_features)
        """

        scale = self.scale_mapper(noise).unsqueeze(-1)
        bias = self.bias_mapper(noise).unsqueeze(-1)

        mu = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)

        x = (x - mu)/std*scale + bias

        return x


class PriceSeriesGenerator(nn.Module):
    """Generative model for simulating a sequences of normalized returns given a historical context
    """
    def __init__(self, noise_dim, n_features):
        super().__init__()

        self.n_emb_feats = 8
        self.n_LCH_feats = 3
        self.noise_dim = noise_dim

        self.embeddings = nn.ModuleDict({
            'month': nn.Embedding(num_embeddings=12, embedding_dim=2),
            'dow': nn.Embedding(num_embeddings=5, embedding_dim=2),
            'trading_day': nn.Embedding(num_embeddings=23, embedding_dim=1),
            'trading_days_left': nn.Embedding(num_embeddings=23, embedding_dim=1),
            'earnings_marker': nn.Embedding(num_embeddings=5, embedding_dim=2)
        })

        self.rnn = nn.GRU(
            input_size=n_features + self.n_emb_feats + noise_dim, 
            hidden_size=n_features//2, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True,
        )

        self.output_mapper = nn.Sequential(
            nn.Linear(n_features, self.n_LCH_feats)
        )

    def _emb_seq(self, x):
        """
        Args:
            x (tensor): Embedding sequence (batch_size, seq_len, 5)
        """
        embs = torch.cat([
            self.embeddings[k](x[:,:,i])
            for i, k 
            in enumerate(['month', 'dow', 'trading_day', 'trading_days_left', 'earnings_marker'])
        ], dim=-1)
        
        return embs

    def gen_noise(self, batch_size, seq_len, mean=0, std=1, clip_lhs=-6, clip_rhs=6, random_state=None, device='cpu'):
        """Generates random, normally distributed noise vectors with truncation
        """
        a, b = (clip_lhs - mean)/std, (clip_rhs - mean)/std
        r = truncnorm.rvs(a, b, size=batch_size*seq_len*self.noise_dim, random_state=random_state)
        r = r.reshape(batch_size, seq_len, self.noise_dim)
        return torch.from_numpy(r).to(torch.float).to(device)

    def forward(self, context, x_emb, noise):
        """

        Args:
            context (tensor): (batch_size, seq_len, n_features)
            noise (tensor): (batch_size, noise_dim)
        """
        batch_size, seq_len, n_features = context.shape

        # Embed inputs and concat to context features
        x = torch.cat((self._emb_seq(x_emb), context, noise), dim=-1)

        # RNN pass
        x, _ = self.rnn(x)

        # Map to output
        x = x.reshape(batch_size*seq_len, n_features)
        x = self.output_mapper(x)
        x = x.reshape(batch_size, seq_len, self.n_LCH_feats)

        return x

    def generate_samples(self, context, x_emb, n_samples):
        """Generate `n_samples` for each item in a batch
        
        Args:
            context (tensor): Context input (batch_size, seq_len, n_features)
            x_emb (tensor): Embedding input (batch_size, seq_len, n_emb_feats)
            n_samples (int): Number of samples to generate for each item in the batch
            
        Returns:
            LCH_samples (tensor): Generated samples (batch_size, n_samples, seq_len, 3)
        """
        
        batch_size, seq_len = x_emb.shape[:2]
        
        # Use the same input noise vectors for each item
        noise = self.gen_noise(n_samples, seq_len)
        
        LCH_samples = []
        for i in range(batch_size):
            samples_i = self.forward(
                torch.repeat_interleave(context[i].unsqueeze(0), repeats=n_samples, dim=0),
                torch.repeat_interleave(x_emb[i].unsqueeze(0), repeats=n_samples, dim=0),
                noise
            )
            LCH_samples.append(samples_i)
            
        LCH_samples = torch.stack(LCH_samples, dim=0)
        
        return LCH_samples


class PriceSeriesCritic(nn.Module):
    """Critic model for distinguishing between real and generated sequences of normalized returns given a historical context
    """
    def __init__(self, n_features):
        super().__init__()

        self.n_emb_feats = 8
        self.n_LCH_feats = 3

        self.embeddings = nn.ModuleDict({
            'month': nn.Embedding(num_embeddings=12, embedding_dim=2),
            'dow': nn.Embedding(num_embeddings=5, embedding_dim=2),
            'trading_day': nn.Embedding(num_embeddings=23, embedding_dim=1),
            'trading_days_left': nn.Embedding(num_embeddings=23, embedding_dim=1),
            'earnings_marker': nn.Embedding(num_embeddings=5, embedding_dim=2)
        })

        self.sequence_featurizer = nn.GRU(
            input_size=n_features + self.n_emb_feats + self.n_LCH_feats, 
            hidden_size=n_features//2, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True,
        )

        # Critic scorer
        self.scorer = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.LeakyReLU(0.2),
            nn.Linear(n_features, 1)
        )

    def _emb_seq(self, x):
        """
        Args:
            x (tensor): Embedding sequence (batch_size, seq_len, 5)
        """
        embs = torch.cat([
            self.embeddings[k](x[:,:,i])
            for i, k 
            in enumerate(['month', 'dow', 'trading_day', 'trading_days_left', 'earnings_marker'])
        ], dim=-1)
        
        return embs

    def forward(self, context, x_emb, x_LCH):
        batch_size, seq_len, n_features = context.shape
        
        # Embed inputs and concat to LCH inputs and context
        x = torch.cat((self._emb_seq(x_emb), x_LCH, context), dim=-1)

        # Featurize sequence
        x, _ = self.sequence_featurizer(x)

        # Map to critic scores
        x = x.reshape(-1, n_features)
        x = self.scorer(x)
        x = x.reshape(batch_size, seq_len)
        
        return x
