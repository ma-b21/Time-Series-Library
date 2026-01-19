import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()
    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    def _get_statistics(self, x):
        dim2reduce = tuple(range(x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        return self.mean, self.stdev
    def normalize(self, x):
        mean, stdev = self._get_statistics(x)
        x = (x - mean) / stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x
    def denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps*self.affine_weight)
        x = x * self.stdev + self.mean
        return x

class Statistic_Features(nn.Module):
    def __init__(self, time_dim, dropout, FFT=1, roll_win=10):
        super().__init__()
        self.roll_win = roll_win
        self.FFT = True if FFT == 1 else False
        self.time_dim = time_dim
        
        if self.FFT:
            self.freq_dim = 4
            self.freq_proj = nn.Linear(96 // 2 + 1, self.freq_dim) 
        else:
            self.freq_dim = 0
            
    def get_statistic_feature(self, data, time_emb=None):
        B_E, C, T = data.size()
        global_mean = torch.mean(data, dim=2, keepdim=True).expand(-1, -1, T)
        
        pad = (self.roll_win - 1) // 2
        avg = F.avg_pool1d(data, kernel_size=self.roll_win, stride=1, padding=pad)
        if avg.shape[2] != T: avg = F.pad(avg, (0, T - avg.shape[2]))
        
        x2 = data.pow(2)
        avg2 = F.avg_pool1d(x2, kernel_size=self.roll_win, stride=1, padding=pad)
        if avg2.shape[2] != T: avg2 = F.pad(avg2, (0, T - avg2.shape[2]))
        std = torch.sqrt(F.relu(avg2 - avg.pow(2)) + 1e-6)
        
        _max = F.max_pool1d(data, kernel_size=self.roll_win, stride=1, padding=pad)
        if _max.shape[2] != T: _max = F.pad(_max, (0, T - _max.shape[2]))
        _min = -F.max_pool1d(-data, kernel_size=self.roll_win, stride=1, padding=pad)
        if _min.shape[2] != T: _min = F.pad(_min, (0, T - _min.shape[2]))
        
        freq_feat = None
        if self.FFT:
            try:
                x_fft = torch.fft.rfft(data, dim=-1, norm='ortho') 
                amp = torch.abs(x_fft) 
                f = self.freq_proj(amp) 
                freq_feat = f.permute(0, 2, 1).repeat(1, 1, T)
            except:
                freq_feat = torch.zeros(B_E, self.freq_dim, T, device=data.device)

        feats = [data, global_mean, avg, std, _max, _min]
        if time_emb is not None: feats.append(time_emb)
        if freq_feat is not None: feats.append(freq_feat)
        return torch.cat(feats, dim=1)

class Full_Temp_Causal_Address(nn.Module):
    def __init__(self, endo_dim, exo_dim, time_dim, k_lookback, roll_win=10, FFT=True, dropout=0.1, seq_len=96):
        super().__init__()
        self.endo_dim = endo_dim
        self.exo_dim = exo_dim
        self.k_lookback = k_lookback
        self.roll_win = roll_win  
        self.dropout = dropout
        self.time_dim = time_dim
        self.FFT = 1 if FFT else 0
        freq_dim = 4 if self.FFT else 0
        
        self.stat_sup_dim = 6 + freq_dim + time_dim 
        
        self.endo_expander = nn.Sequential(
            nn.Conv1d(self.stat_sup_dim, self.exo_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.stat_feature_extractor = Statistic_Features(self.time_dim, self.dropout, self.FFT, self.roll_win)
        self.stat_feature_extractor.freq_proj = nn.Linear(seq_len // 2 + 1, freq_dim)
        
    def endo_augmentation(self, endo, x_mark_enc):
        stat_sup_endo = self.stat_feature_extractor.get_statistic_feature(endo, x_mark_enc) 
        augmentated_endo = self.endo_expander(stat_sup_endo) 
        return augmentated_endo
    
    def forward(self, endo, x_mark_enc, exo=None, x_mark_exo=None):
        augmentated_endo = self.endo_augmentation(endo, x_mark_enc) 
        
        if exo is not None:
            # MS模式：用 Target 去寻址 Auxiliary 
            augmentated_exo = self.endo_augmentation(exo, x_mark_exo)
            B_endo = augmentated_endo.shape[0]
            B_exo = augmentated_exo.shape[0]

            if B_exo > B_endo:
                num_aux = B_exo // B_endo
                target_ref = augmentated_exo.view(B_endo, num_aux, self.exo_dim, -1).mean(dim=1)
            else:
                target_ref = augmentated_exo
        else:
            # M模式/单变量：自寻址
            target_ref = augmentated_endo 
        
        target_padded = F.pad(target_ref, (self.k_lookback-1, 0)) 
        target_windows = target_padded.unfold(dimension=2, size=self.k_lookback, step=1)
        
        product = augmentated_endo.unsqueeze(-1) * target_windows 
        score = F.softmax(product, dim=-1)
        exo_emb = (score * target_windows).sum(dim=-1) 
        
        return augmentated_endo, exo_emb

class VectorizedCausalProjection(nn.Module):
    def __init__(self, query_dim, key_dim, k_lookback=16, dropout=0.3):
        super().__init__()
        self.k = k_lookback
        self.gen_input_dim = query_dim * 6 
        self.out_channels = key_dim * k_lookback 
        self.kernel_generator = nn.Conv1d(self.gen_input_dim, self.out_channels, kernel_size=1)
        nn.init.xavier_normal_(self.kernel_generator.weight, gain=0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_feat, context_block):
        BM, E, T = query_feat.shape
        K = self.k
        g_mean = torch.mean(query_feat, dim=2, keepdim=True).expand(-1, -1, T)
        x_squared = torch.pow(query_feat, 2)
        avg_x = F.avg_pool1d(query_feat, kernel_size=5, stride=1, padding=2)
        avg_x2 = F.avg_pool1d(x_squared, kernel_size=5, stride=1, padding=2)
        rolling_var = F.relu(avg_x2 - torch.pow(avg_x, 2)) + 1e-6
        rolling_std = torch.sqrt(rolling_var)
        third_moment = torch.pow(query_feat - avg_x, 3)
        rolling_max = F.max_pool1d(query_feat, kernel_size=5, stride=1, padding=2)
        rolling_min = -F.max_pool1d(-query_feat, kernel_size=5, stride=1, padding=2)
        
        gen_in = torch.cat([self.dropout(query_feat), g_mean, rolling_std, third_moment, rolling_max, rolling_min], dim=1)
        kernels = self.kernel_generator(gen_in).view(BM, E, K, T)
        ctx_padded = F.pad(context_block, (K-1, 0))
        ctx_windows = ctx_padded.unfold(dimension=2, size=K, step=1).permute(0, 1, 3, 2)
        attn_weights = F.softmax(kernels, dim=2)
        y_conservative = (ctx_windows * attn_weights).sum(dim=2)
        return y_conservative

class GatedExoProjector(nn.Module):
    def __init__(self, n_vars, d_model=32, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.var_projector = nn.Sequential(nn.Linear(n_vars, d_model), nn.LeakyReLU(), nn.Dropout(dropout))
        self.out_proj = nn.Linear(d_model, n_vars)
        self.channel_gate = nn.Parameter(torch.zeros(n_vars)) 
    def forward(self, pred_ci):
        B, T, M = pred_ci.shape
        x_flat = pred_ci.reshape(B*T, M) 
        global_ctx = self.var_projector(x_flat) 
        correction_flat = self.out_proj(global_ctx)
        correction = correction_flat.reshape(B, T, M)
        gate = torch.tanh(self.channel_gate).view(1, 1, M)
        return correction * gate

class Lite_Temp_Causal_Address(nn.Module):
    def __init__(self, d_model, k_lookback, dropout=0.1):
        super().__init__()
        self.k_lookback = k_lookback
        self.d_model = d_model
        
        self.feature_proj = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gate = nn.Parameter(torch.tensor(0.0)) 

    def forward(self, x):
        query = self.feature_proj(x)
        
        # Self-Addressing within Raw Features
        keys_padded = F.pad(query, (self.k_lookback-1, 0))
        keys_windows = keys_padded.unfold(dimension=2, size=self.k_lookback, step=1)
        
        product = query.unsqueeze(-1) * keys_windows 
        energy = product.sum(dim=1) 
        attn_scores = F.softmax(energy / (self.d_model ** 0.5), dim=-1)
        
        attn_scores_expanded = attn_scores.unsqueeze(1) 
        context = (attn_scores_expanded * keys_windows).sum(dim=-1)
        
        return context * torch.tanh(self.gate)
