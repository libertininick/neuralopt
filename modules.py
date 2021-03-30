
from scipy.stats import truncnorm
import torch
import torch.nn as nn


def _init_linear(layer, mean=0, std=0.01, bias=0):
    layer.weight.data = nn.init.normal_(layer.weight.data, mean=mean, std=std) 
    if layer.bias is not None: 
        layer.bias.data = nn.init.constant_(layer.bias.data, bias)


class AttentionHead(nn.Module):
    def __init__(self, in_features, out_features, dropout=0):
        """Applies query, key, value matrix transformations to input features.
        
        Args:
            in_features (int): Number of input features. 
            out_features (int): Number of output features after attention mixing
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

        # Softmax
        self.softmax = nn.Softmax(dim=-1)
        
        self._init_wts()
        
    def _init_wts(self):
        for m in [self.query, self.key, self.value]:
            m.weight.data = nn.init.normal_(m.weight.data, mean=0, std=(1/m.in_features)**0.5)

    def get_attention_weights(self, x):
        # Reshape x for query and key transforms
        batch_size, n_obs, n_feats = x.shape
        x = x.reshape(-1, n_feats)
        
        # Calculate query/key vectors
        queries = self.query(x).view(batch_size, n_obs, self.n_out)
        keys = self.key(x).view(batch_size, n_obs, self.n_out)
        
        # Calculate attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2).contiguous())/self.n_out**0.5
        
        # Calculate softmax of scores 
        weights = self.softmax(scores.view(-1, n_obs)).view(batch_size, n_obs, n_obs)
        
        return weights
    
    def forward(self, x):
        """
        x : (batch_size, n_obs, n_feats)
        """
        batch_size, n_obs, n_feats = x.shape
        x = self.dropout(x)

        # Calculate attention weights
        weights = self.get_attention_weights(x)
        
        # Calculate value vectors 
        values = self.value(x.reshape(-1, n_feats)).reshape(batch_size, n_obs, self.n_out) 
         
        # Avg weighted avg values based on attention weights 
        values = torch.bmm(weights, values)

        return values


class EncoderBlock(nn.Module):
    def __init__(
        self, 
        n_features, 
        conv_kernel_size, 
        n_attention_heads,
        rnn_kernel_size,
        negative_slope=0.1,
        dropout=0):
        super().__init__()

        self.negative_slope = negative_slope
        self.cov_feature_extractor = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(
                in_channels=n_features, 
                out_channels=n_features, 
                kernel_size=(conv_kernel_size, 1), 
                padding=(conv_kernel_size//2, 0)
            ),
            nn.LeakyReLU(negative_slope),
        )

        attn_dim = n_features//n_attention_heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                AttentionHead(n_features, attn_dim, dropout),
                nn.LeakyReLU(negative_slope),
                nn.LayerNorm(normalized_shape=attn_dim),
            )
            for _
            in range(n_attention_heads)
        ])

        self.rnn_kernel_size = rnn_kernel_size
        self.rnn = nn.GRU(
            input_size=n_features, 
            hidden_size=n_features//2, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self._init_wts()

    def _init_wts(self):
        for m in self.cov_feature_extractor:
            if isinstance(m, nn.Conv2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data, a=self.negative_slope, mode='fan_in', nonlinearity='relu')

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                param.data = nn.init.normal_(param.data, mean=0, std=0.001) 
            if 'bias' in name:
                param.data = nn.init.constant_(param.data, 3)
    
    def forward(self, x):
        """
        Args:
            x (tensor): (batch_size, seq_len, n_features)

        Returns
            x (tensor): (bath_size, seq_len, n_features)
        """

        # Convolutional feature extraction
        x = x.permute(0, 2, 1).unsqueeze(-1) # Move features to channels dim and add phantom width dim
        x = self.cov_feature_extractor(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        
        # Attention mixing + skip connection
        a = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        x = x + a

        # RNN pass
        # h represents the lass hidden state of each of the rnn_kernel windows
        batch_size, seq_len, n_features = x.shape
        o, h = self.rnn(x.reshape(-1, self.rnn_kernel_size, n_features))
        x = x + o.reshape(batch_size, seq_len, n_features)
        h = h.reshape(
            self.rnn.num_layers*(self.rnn.bidirectional + 1), # num_layers * num_directions
            batch_size,                                       # original batch size
            n_features//2,                                    # hidden size
            seq_len//self.rnn_kernel_size                     # number of distinct rnn_kernel windows
        )

        return x, h


class PriceSeriesFeaturizer(nn.Module):
    def __init__(
        self, 
        n_features,
        historical_seq_len,
        future_seq_len,
        n_dist_targets,
        conv_kernel_size, 
        n_attention_heads,
        rnn_kernel_size,
        n_blocks, 
        dropout=0):

        super().__init__()

        self.historical_seq_len = historical_seq_len
        self.future_seq_len = future_seq_len

        self.n_emb_feats = 8
        self.n_LCH_feats = 3
        self.pos_emb_dim = int(future_seq_len**0.5)
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

        # Input upsampler
        self.input_upsampler = nn.Linear(
            in_features=self.n_emb_feats + self.n_LCH_feats, 
            out_features=n_features
        )
        
        # Historical sequence encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(n_features, conv_kernel_size, n_attention_heads, rnn_kernel_size, dropout=dropout)
            for _
            in range(n_blocks)
        ])

        # Historical feature averager
        self.h_feat_averager = nn.parameter.Parameter(
            data=torch.randn(future_seq_len, historical_seq_len)*0.01
        )
        self.softmax = nn.Softmax(dim=-1)

        # Future position indxes
        self.pos_idxs = torch.arange(future_seq_len).unsqueeze(0)
        
        # Feature sequence encoder
        self.future_seq_encoder = nn.GRU(
            input_size=self.n_emb_feats + self.pos_emb_dim + n_features,  
            hidden_size=n_features//2,  
            num_layers=2,  
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

        # Sequence vectorizer
        self.seq_vectorizer = nn.GRU(
            input_size=n_features,  
            hidden_size=n_features//4,  
            num_layers=2,  
            batch_first=True, 
            bidirectional=True, 
        )

        # Sequence order classifier
        self.seq_order_classifier = nn.Sequential(
            nn.Linear(n_features*2, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_wts()

    def _init_wts(self):
        for emb in self.embeddings.values():
            emb.weight.data = nn.init.normal_(emb.weight.data, mean=0, std=0.01)

        for module in [
            self.input_upsampler,
            self.LCH_mapper,
            self.dists_forecaster,
            self.seq_order_classifier,
        ]:

            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        _init_linear(layer)
            elif isinstance(module, nn.Linear):
                _init_linear(module)

        for module in [
            self.future_seq_encoder,
            self.seq_vectorizer,
        ]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                if 'bias' in name:
                    nn.init.zeros_(param)

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

        batch_size, seq_len = x_emb.shape[:2]

        # Embed inputs and concat to float inputs
        x = torch.cat((self._emb_seq(x_emb), x_LCH), dim=-1)

        # Upsample input to n_features
        x = x.reshape(-1, self.n_emb_feats + self.n_LCH_feats)
        x = self.input_upsampler(x)
        x = x.reshape(batch_size, seq_len, self.n_features)

        # Encode with skip connections
        for block in self.encoder_blocks:
            o, _ = block(x)
            x = x + o

        return x

    def encode_decode(self, h_emb, h_LCH):

        batch_size, seq_len, _ = h_emb.shape

        # Encode
        x = self.encode(h_emb, h_LCH)

        # Map input to reconstruction LCH features
        x = x.reshape(-1, self.n_features)
        x = self.LCH_mapper(x)
        x = x.reshape(batch_size, seq_len, self.n_LCH_feats)
        
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

        # Init Future features
        batch_size = f_emb.shape[0]
        f_emb = self._emb_seq(f_emb)
        f_pos_enc = self._encode_position(batch_size)
        f_feat = torch.cat([f_emb, f_pos_enc, h_avg], dim=-1)

        # Encode
        f_feat, _ = self.future_seq_encoder(f_feat)

        return f_feat

    def vectorize_seq(self, x_emb, x_LCH):
        """Return a vector representing a price sequence
        """

        # Encode sequence
        seq_feat = self.encode(x_emb, x_LCH)

        # Vectorize
        seq_feat, _ = self.seq_vectorizer(seq_feat)
        seq_vectors = torch.cat((seq_feat[:,0,:], seq_feat[:,-1,:]), dim=-1)

        return seq_vectors

    def forecast(self, h_emb, h_LCH, f_emb):
        batch_size = f_emb.shape[0]

        # Encode future path
        f_feat = self.encode_future_path(h_emb, h_LCH, f_emb)
        f_feat = f_feat.reshape(batch_size*self.future_seq_len, self.n_features)

        # Future LCH seq predictions
        # f_LCH = self.LCH_forecaster(f_feat).reshape(batch_size, self.future_seq_len, 3)
        f_LCH = self.LCH_mapper(f_feat).reshape(batch_size, self.future_seq_len, 3)

        # Probability predictions
        f_ret_probas = self.dists_forecaster(f_feat).reshape(batch_size, self.future_seq_len, self.n_dist_targets)

        return f_LCH, f_ret_probas


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
    def __init__(self, noise_dim, seq_len, n_features, n_out, n_attention_heads, negative_slope=0.1, dropout=0):
        super().__init__()

        self.noise_dim = noise_dim

        self.noise_mapper = nn.Sequential(
            nn.Linear(noise_dim, 4*n_features),
            nn.LeakyReLU(negative_slope),
            nn.Linear(4*n_features, 2*n_features),
            nn.LeakyReLU(negative_slope),
            nn.Linear(2*n_features, n_features),
        )

        self.ada_in_1 = AdaptiveInstanceNormalization(seq_len, n_features)
        self.ada_in_2 = AdaptiveInstanceNormalization(seq_len, n_features)

        attn_dim = n_features//n_attention_heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                AttentionHead(n_features, attn_dim, dropout),
                nn.LeakyReLU(negative_slope),
                nn.LayerNorm(normalized_shape=attn_dim),
            )
            for _
            in range(n_attention_heads)
        ])

        self.noise_to_hidden_state_mapper = nn.Sequential(
            nn.Linear(n_features, 2*2*(n_features//2)),  # (num_layers*num_directions, batch, hidden_size)
            nn.Tanh(),
        )
        self.rnn = nn.GRU(
            input_size=n_features, 
            hidden_size=n_features//2, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.output_mapper = nn.Linear(n_features, n_out)

    def gen_noise(self, batch_size, mean=0, std=1, clip_lhs=-6, clip_rhs=6, random_state=None, device='cpu'):
        """Generates random, normally distributed noise vectors with truncation
        """
        a, b = (clip_lhs - mean)/std, (clip_rhs - mean)/std
        r = truncnorm.rvs(a, b, size=self.noise_dim*batch_size, random_state=random_state)
        r = r.reshape(batch_size, self.noise_dim)
        return torch.from_numpy(r).to(torch.float).to(device)

    def forward(self, context, noise):
        """

        Args:
            context (tensor): (batch_size, seq_len, n_features)
            noise (tensor): (batch_size, noise_dim)
        """
        batch_size, seq_len, n_features = context.shape

        # Map noise and add to context
        noise = self.noise_mapper(noise)
        x = context + noise.unsqueeze(1)

        # Self attention + skip connection
        a = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        x = x + a

        # Adaptive instance normalization
        x = self.ada_in_1(x, noise)

        # RNN pass
        h0 = self.noise_to_hidden_state_mapper(noise).reshape(2*2, batch_size, n_features//2)
        o, _ = self.rnn(x, h0)
        x = x + o

        # Adaptive instance normalization
        x = self.ada_in_2(x, noise)

        # Map to output
        x = x.reshape(batch_size*seq_len, n_features)
        x = self.output_mapper(x)
        x = x.reshape(batch_size, seq_len, -1)

        return x


class PriceSeriesCritic(nn.Module):
    """Critic model for distinguishing between real and generated sequences of normalized returns given a historical context
    """
    def __init__(
        self, 
        n_features, 
        conv_kernel_size, 
        n_attention_heads,
        rnn_kernel_size,
        n_blocks, 
        dropout=0):
        super().__init__()

        # Upsampling mapper
        self.seq_upsampler = nn.utils.spectral_norm(nn.Linear(in_features=3, out_features=n_features))
        
        # Encoder blocks
        self.blocks = nn.ModuleList([
            CriticEncoderBlock(n_features*2, conv_kernel_size, n_attention_heads, rnn_kernel_size, dropout=dropout)
            for _
            in range(n_blocks)
        ])

        # Critic scorer
        self.scorer = nn.utils.spectral_norm(nn.Linear(n_features*2, 1))
        
        self._init_wts()

    def _init_wts(self):

        for module in [self.seq_upsampler, self.scorer]:

            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        _init_linear(layer)
            elif isinstance(module, nn.Linear):
                _init_linear(module)

    def forward(self, context, LCH_seq):
        batch_size, seq_len, n_features = context.shape
        
        # Upsample low, close, high sequence
        seq = LCH_seq.reshape(batch_size*seq_len, -1)
        seq = self.seq_upsampler(seq)
        seq = seq.reshape(batch_size, seq_len, n_features)

        # Combine context and sequence
        x = torch.cat((context, seq), dim=-1)

        # Encode with skip connections
        for block in self.blocks:
            o, _ = block(x)
            x = x + o

        # Map encodings to critic scores
        x = x.reshape(-1, n_features*2)
        x = self.scorer(x)
        x = x.reshape(batch_size, seq_len)
        
        return x
