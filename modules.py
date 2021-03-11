import numpy as np
import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, in_features, out_features, dropout):
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
        self.query_key = nn.Linear(in_features, out_features, bias=False)
        self.value = nn.Linear(in_features, out_features, bias=False)

        # Feed-forward layer
        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(out_features)
        )

        # Softmax
        self.softmax = nn.Softmax(dim=-1)
        
        self._init_wts()
        
    def _init_wts(self):
        for m in [self.query_key, self.value]:
            m.weight.data = nn.init.normal_(m.weight.data, mean=0, std=(1/m.in_features)**0.5)

    def get_attention_weights(self, x):
        # Reshape x for query and key transforms
        batch_size, n_obs, n_feats = x.shape
        x = x.reshape(-1, n_feats)
        
        # Calculate query/key vectors
        queries_keys = self.query_key(x).view(batch_size, n_obs, self.n_out)
        
        # Calculate attention scores
        scores = torch.bmm(queries_keys, queries_keys.transpose(1, 2).contiguous())/self.n_out**0.5
        
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

        # Feed forward
        output = self.ffn(values.reshape(-1, self.n_out)).reshape(batch_size, n_obs, self.n_out)

        return output


class EncoderBlock(nn.Module):
    def __init__(self, n_attn_heads, in_channels, out_channels, stride, dropout):
        super().__init__()
        
        attn_dim = in_channels//n_attn_heads
        self.attention_heads = nn.ModuleList([
            AttentionHead(in_channels, attn_dim, dropout)
            for _
            in range(n_attn_heads)
        ])

        self.strided_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(stride,1), stride=stride),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=out_channels),
        )
            
    def forward(self, x):
        """
        Args:
            x (tensor): (batch_size, n_channels, seq_len, 1)
        """
        
        # Attention mixing
        x = x.squeeze(-1).permute(0,2,1)
        x = torch.cat([
            head(x)
            for head
            in self.attention_heads
        ], dim=-1)
        x = x.permute(0,2,1).unsqueeze(-1)
        
        # Strided convolution compression
        x = self.strided_conv(x)
        
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_attn_heads, in_channels, out_channels, stride, dropout):
        super().__init__()
        
        self.strided_transpose_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(stride,1), stride=stride),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=out_channels),
        )
        
        attn_dim = in_channels//n_attn_heads
        self.attention_heads = nn.ModuleList([
            AttentionHead(in_channels, attn_dim, dropout)
            for _
            in range(n_attn_heads)
        ])
            
    def forward(self, x):
        """
        Args:
            x (tensor): (batch_size, n_channels, seq_len, 1)
        """
        
        # Stided convolution reconstruction
        x = self.strided_transpose_conv(x)
        
        # Attention mixing
        x = x.squeeze(-1).permute(0,2,1)
        x = torch.cat([
            head(x)
            for head
            in self.attention_heads
        ], dim=-1)
        x = x.permute(0,2,1).unsqueeze(-1)
        
        return x


class PriceSeriesFeaturizer(nn.Module):
    """Encodes a daily price series into a fixed length vector
    """
    def __init__(
        self, 
        n_historical,
        n_forecast_targets,
        encoding_dim,
        n_attn_heads=4,
        compression_stride=4, 
        layer_dropout=0.1, 
        cross_connect_dropout=0.25):

        super().__init__()
        
        self.embeddings = nn.ModuleDict({
            'month': nn.Embedding(num_embeddings=12, embedding_dim=2),
            'dow': nn.Embedding(num_embeddings=5, embedding_dim=2),
            'trading_day': nn.Embedding(num_embeddings=23, embedding_dim=1),
            'trading_days_left': nn.Embedding(num_embeddings=23, embedding_dim=1)
        })
        
        # Encoder
        n_layers = int(np.log(n_historical)/np.log(compression_stride))
        encoder_layers = [
            nn.Sequential(
                nn.Conv2d(in_channels=11, out_channels=encoding_dim, kernel_size=(7,1), padding=(7//2,0)),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(num_features=encoding_dim),
            )
        ]
        encoder_layers += [EncoderBlock(n_attn_heads, encoding_dim, encoding_dim, compression_stride, layer_dropout)]*n_layers
        self.encoder = nn.ModuleList(encoder_layers)

        # Decoder
        self.cross_connection_dropout = nn.Dropout2d(cross_connect_dropout)
        decoder_layers = [DecoderBlock(n_attn_heads, encoding_dim, encoding_dim, compression_stride, layer_dropout)]*n_layers
        decoder_layers += [
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=5, kernel_size=(7,1), padding=(7//2,0)),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(num_features=5),
            )
        ]
        self.decoder = nn.ModuleList(decoder_layers)

        # Forecaster
        self.rnn_1 = nn.GRU(
            input_size=6, 
            hidden_size=encoding_dim//2, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True,
            dropout=layer_dropout
        )
        self.rnn_2 = nn.GRU(
            input_size=encoding_dim, 
            hidden_size=encoding_dim//2, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True,
            dropout=layer_dropout
        )
        self.target_classifier = nn.Linear(encoding_dim, n_forecast_targets)


    def emb_seq(self, x):
        """
        Args:
            x (tensor): Embedding sequence (batch_size, seq_len, 4)
        """
        embs = torch.cat([
            self.embeddings[k](x[:,:,i])
            for i, k 
            in enumerate(['month', 'dow', 'trading_day', 'trading_days_left'])
        ], dim=-1)
        
        return embs
    
    def encode(self, x_emb, x_float):
        x = torch.cat((self.emb_seq(x_emb), x_float), dim=-1)
        x = x.permute(0,2,1).unsqueeze(-1)
        
        for layer in self.encoder:
            x = layer(x)

        return x.squeeze()

    def encode_decode(self, x_emb, x_float):
        x = torch.cat((self.emb_seq(x_emb), x_float), dim=-1)
        x = x.permute(0,2,1).unsqueeze(-1)
        
        layer_states = []
        for layer in self.encoder:
            x = layer(x)
            layer_states.append(x)
        
        for layer, state in zip(self.decoder[:-1], layer_states[:-1][::-1]):
            x = layer(x) + self.cross_connection_dropout(state)
        x = self.decoder[-1](x)
        
        x = x.squeeze(-1).permute(0,2,1)
        
        return x

    def get_forecast_context(self, h_emb, h_float, f_emb):
        """Context embedding for each period in forecast window
        """
        # Encode historical price series
        h_encoding = self.encode(h_emb, h_float)
        h_encoding = h_encoding.unsqueeze(dim=1)
        
        # Embed forecast window
        f_seq = self.emb_seq(f_emb)
        f_seq, _ = self.rnn_1(f_seq)

        # Add historical encoding to each period
        f_seq = f_seq + h_encoding

        # Generate context
        f_context, _ = self.rnn_2(f_seq)

        return f_context

    def forecast(self, h_emb, h_float, f_emb):
        """Target predictions for each period in forecast window"""
        
        f_context = self.get_forecast_context(h_emb, h_float, f_emb)

        batch_size, seq_len, n_feat = f_context.shape
        f_context = f_context.reshape(-1, n_feat)
        f_probas = self.target_classifier(f_context)
        f_probas = torch.cumsum(f_probas, dim=-1)  # Cumulative probability across target thresholds
        f_probas = f_probas.reshape(batch_size, seq_len, -1)
        f_probas = torch.sigmoid(f_probas)

        return f_probas