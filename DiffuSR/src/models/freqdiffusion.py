import torch.nn as nn
import torch as th
from matplotlib import pyplot as plt

from .step_sample import create_named_schedule_sampler
import numpy as np
import math
import torch
import torch.nn.functional as F


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """

    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar in the limit of num_diffusion_timesteps. Beta schedules may be added, but should not be removed or changed once they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps,
                                   lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2, )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: 1 - np.sqrt(t + 0.0001), )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(num_diffusion_timesteps, lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2, )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  # scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10, dtype=np.float64)
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and produces the cumulative product of (1-beta) up to that part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):  ## 2000
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1 - alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps - 1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (
                1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in
                   zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1, 1, 1, corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e4)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden,
                                     lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Transformer_rep(nn.Module):
    def __init__(self, args):
        super(Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)])

    def forward(self, hidden, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        return hidden


class Diffu_xstart(nn.Module):
    def __init__(self, hidden_size, args, max_seq_length=50):
        super(Diffu_xstart, self).__init__()
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.linear_item = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_xt = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size)
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim), SiLU(),
                                        nn.Linear(time_embed_dim, self.hidden_size))
        self.fuse_linear = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.att = Transformer_rep(args)
        # self.mlp_model = nn.Linear(self.hidden_size, self.hidden_size)
        # self.gru_model = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # self.gru_model = nn.GRU(self.hidden_size, self.hidden_size, num_layers=args.num_blocks, batch_first=True)
        self.lambda_uncertainty = args.lambda_uncertainty
        self.dropout = nn.Dropout(args.dropout)
        self.norm_diffu_rep = LayerNorm(self.hidden_size)
        self.freq_filer = FreqFilterLayer(
            hidden_size=self.hidden_size,
            max_seq_len=self.max_seq_length,
            init_s1=0.25, init_s2=0.5,
            k=20.0,
            norm='ortho'
        )

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(
            device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, x_t, t, mask_seq):
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        emb_t = emb_t.unsqueeze(1).expand(-1, x_t.size(1), -1)
        x_t = x_t + emb_t

        # freq_code = torch.sin(t[:, None] / 2000 * math.pi).unsqueeze(1).expand_as(x_t)
        # x_t = x_t + 0.1 * freq_code

        # lambda_uncertainty = th.normal(mean=th.full(rep_item.shape, 1.0), std=th.full(rep_item.shape, 1.0)).to(x_t.device)

        lambda_uncertainty = th.normal(
            mean=th.full(rep_item.shape, self.lambda_uncertainty, device=x_t.device),
            std=th.full(rep_item.shape, self.lambda_uncertainty, device=x_t.device)
        )
        ## distribution
        # lambda_uncertainty = self.lambda_uncertainty  ### fixed

        ####  Attention

        rep_item = self.freq_filer(rep_item)
        # x_t = self.freq_filer(x_t)

        rep_diffu = self.att(rep_item + lambda_uncertainty * x_t, mask_seq)
        rep_diffu = self.norm_diffu_rep(self.dropout(rep_diffu))
        out = rep_diffu[:, -1, :]

        return out, rep_diffu


class FreqFilterLayer(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            max_seq_len: int,
            init_s1: float = 0.25,
            init_s2: float = 0.5,
            init_tau: float = 1e-3,
            k: float = 8.0,
            beta: float = 6.0,
            norm: str = 'ortho',
            hidden_dropout_prob=0.5,
            shared_s1: nn.Parameter = None,
            shared_s2: nn.Parameter = None,
    ):
        super().__init__()
        assert 0.0 < init_s1 < init_s2 < 1.0, "require 0 < s1 < s2 < 1"
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.norm = norm
        self.k = k
        self.beta = beta
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        # self.gate_mlp = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size // 2, 3),
        #     nn.Softmax(dim=-1)
        # )

        self.band_attn = FreqBandAttention(hidden_size=self.hidden_size, num_heads=4)

        self.raw_s1 = shared_s1
        self.raw_s2 = shared_s2

        # tau 可学习
        self.tau = nn.Parameter(torch.tensor(init_tau, dtype=torch.float32))

        self.ln = nn.LayerNorm(hidden_size, eps=1e-12)
        self.res_gate = nn.Parameter(torch.tensor(0.3))  # 在 __init__ 里添加

        # 缓存频率刻度 [0,1]，使s12是和序列长度无关的比例参数
        rfft_len = max_seq_len // 2 + 1
        u = torch.linspace(0.0, 1.0, rfft_len)
        self.register_buffer("u_grid", u)

    def _soft_band_masks(self, s1, s2):
        """三段软掩码"""
        u = self.u_grid
        low_mask = torch.sigmoid(self.k * (s1 - u))
        high_mask = torch.sigmoid(self.k * (u - s2))
        left = torch.sigmoid(self.k * (u - s1))
        right = torch.sigmoid(self.k * (s2 - u))
        band_mask = left * right
        return (low_mask[None, :, None],
                band_mask[None, :, None],
                high_mask[None, :, None])

    def forward(self, x):
        """
        x: [B, T, H]
        """
        B, T, H = x.shape
        assert T <= self.max_seq_len

        X = torch.fft.rfft(x, dim=1, norm=self.norm)

        # 幅值软阈过滤 (tau 为可学习参数，确保非负)
        tau_val = torch.clamp(self.tau, min=1e-6)
        mag = torch.abs(X)
        keep = torch.sigmoid(self.beta * (mag - tau_val))
        X = X * keep

        seq_repr = x.mean(dim=1)
        # gate = self.gate_mlp(seq_repr)

        # w_low, w_band, w_high = gate[:, 0], gate[:, 1], gate[:, 2]

        # w_low = w_low[:, None, None]
        # w_band = w_band[:, None, None]
        # w_high = w_high[:, None, None]

        # 计算 s1, s2
        s1 = torch.sigmoid(self.raw_s1)
        s2 = torch.sigmoid(self.raw_s2)
        s1 = torch.clamp(s1, 1e-3, 1 - 2e-3)
        s2 = torch.clamp(s2, s1 + 1e-3, torch.tensor(0.999, device=s2.device))

        low_m, band_m, high_m = self._soft_band_masks(s1, s2)

        # 3) 分别回到时域得到三条分量
        x_low = torch.fft.irfft(X * low_m, n=T, dim=1, norm=self.norm)  # [B,T,H]
        x_mid = torch.fft.irfft(X * band_m, n=T, dim=1, norm=self.norm)  # [B,T,H]
        x_high = torch.fft.irfft(X * high_m, n=T, dim=1, norm=self.norm)

        bands_three = torch.stack([x_low, x_mid, x_high], dim=2)

        x_attn = self.band_attn(x, bands_three)

        out = self.ln(self.out_dropout(x_attn) + self.res_gate * x)

        return out


class FreqBandAttention(nn.Module):
    def __init__(self, hidden_size, attn_dim=None, num_heads=4, dropout=0.1):
        super().__init__()
        d = attn_dim or hidden_size
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        assert d % num_heads == 0

        self.q_proj = nn.Linear(hidden_size, d)
        self.k_proj = nn.Linear(hidden_size, d)
        self.v_proj = nn.Linear(hidden_size, d)
        self.o_proj = nn.Linear(d, hidden_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_tok, bands_three):
        B, L, _, H = bands_three.shape
        d = self.num_heads * self.head_dim

        Q = self.q_proj(x_tok)
        K = self.k_proj(bands_three)
        V = self.v_proj(bands_three)

        Q = Q.view(B, L, self.num_heads, self.head_dim)
        K = K.view(B, L, 3, self.num_heads, self.head_dim)
        V = V.view(B, L, 3, self.num_heads, self.head_dim)

        scores = (Q.unsqueeze(2) * K).sum(-1) / math.sqrt(self.head_dim)  # [B,T,3,h]
        attn = torch.softmax(scores, dim=2)  # [B,T,3,h]
        attn = self.dropout(attn)

        # 加权汇聚 V
        out = (attn.unsqueeze(-1) * V).sum(dim=2)  # [B,T,h,dh]
        out = out.reshape(B, L, d)  # [B,T,d]
        return self.o_proj(out)


class FreqNoiseGenerator(nn.Module):
    def __init__(self, seq_len, device, shared_s1=None, shared_s2=None,
                 k=10.0, smooth_sched: str = "cosine"):
        super().__init__()
        self.seq_len = seq_len
        self.device = device
        self.k = k
        self.smooth_sched = smooth_sched.lower()
        self.shared_s1 = shared_s1
        self.shared_s2 = shared_s2

        self.raw_start = nn.Parameter(torch.tensor(-1.5))  # [-3,0)
        self.raw_end = nn.Parameter(torch.tensor(1.5))  # (0,3]
        self.raw_tau = nn.Parameter(torch.tensor(0.0))

        # 跟踪参数
        self.param_log = []

        # 预存频率刻度
        self.register_buffer("u_grid", torch.linspace(0, 1, seq_len // 2 + 1))

    # -------- 计算三段掩码 --------
    def _make_masks(self):
        s1 = torch.sigmoid(self.shared_s1)
        s2 = torch.sigmoid(self.shared_s2)
        s1 = torch.clamp(s1, 1e-3, 0.998)
        s2 = torch.clamp(s2, s1 + 1e-3, torch.tensor(0.999, device=s2.device))

        u = self.u_grid.to(s1.device)
        low_mask = torch.sigmoid(self.k * (s1 - u))
        high_mask = torch.sigmoid(self.k * (u - s2))
        mid_mask = torch.sigmoid(self.k * (u - s1)) * torch.sigmoid(self.k * (s2 - u))
        return low_mask, mid_mask, high_mask

    def _smooth_weights(self, t, T):
        rho = t.float() / T  # [B]
        if self.smooth_sched == "cosine":
            # 前期低频多，后期高频多
            a = torch.sin(rho * math.pi / 2) ** 2
            c = (torch.cos(rho * math.pi / 2) ** 2) ** 1.8
            b = torch.exp(-((rho - 0.5) ** 2) / 0.2)
        elif self.smooth_sched == "linear":
            a = rho
            c = 1 - rho
            b = torch.ones_like(a) * 0.3
        else:
            raise ValueError(f"Unknown schedule type: {self.smooth_sched}")

        norm = a + b + c + 1e-8
        return a / norm, b / norm, c / norm

<<<<<<< Updated upstream
    # def get_lambda(self, t, T, delta=0.5):
    #     return torch.sigmoid((t.float() - 0.5 * T) / (delta * T))
    def get_lambda(self, t, T, L, start=-2.54, end=2.30, tau=6.56):
=======
    def get_lambda(self, t, T, L, start=0.0, end=3.0, tau=0.2):
>>>>>>> Stashed changes
        ratio = t.float() / T

        start = -3.0 + 3.0 * torch.sigmoid(self.raw_start)
        end = 3.0 * torch.sigmoid(self.raw_end)
        tau = 0.01 * (1000.0 / 0.01) ** torch.sigmoid(self.raw_tau)

        linear_term = start + (end - start) * ratio
        lam = torch.sigmoid(linear_term / tau)
        self.param_log.append({
            "start": float(start.item()),
            "end": float(end.item()),
            "tau": float(tau.item())
        })

        # self.param_log.append({
        #     "start": start,
        #     "end": end,
        #     "tau": tau
        # })
        return lam

    def forward(self, t_tensor, shape):
        B, L, H = shape
        device = t_tensor.device
        T = 32

        # (1) 批量生成 eps [B, L, H]
        eps = torch.randn((B, L, H), device=device)

        low_mask, band_mask, high_mask = self._make_masks()  # [F]
        F = L // 2 + 1
        low_mask = low_mask[:F]
        band_mask = band_mask[:F]
        high_mask = high_mask[:F]

        # (2) 计算每个样本对应的 a,b,c （向量化）
        a, b, c = self._smooth_weights(t_tensor, T)  # [B]
        a = a[:, None, None]
        b = b[:, None, None]
        c = c[:, None, None]

        power = (
                a * (low_mask ** 2).mean() +
                b * (band_mask ** 2).mean() +
                c * (high_mask ** 2).mean()
        )
        match = (1.0 / (power.sqrt() + 1e-6)).detach()

        # (3) 预先构建 sqrtD 向量 [1, L//2+1, 1]
        sqrtD = match * (
                a * low_mask[None, :, None] ** 2
                + b * band_mask[None, :, None] ** 2
                + c * high_mask[None, :, None] ** 2
        ).sqrt()

        # (4) 对整个 batch 同时进行 FFT [B, L//2+1, H]
        freq = torch.fft.rfft(eps, dim=1)
        freq = sqrtD * freq  # 广播乘法，频域滤波
        noise_i = torch.fft.irfft(freq, n=L, dim=1)  # [B, L, H]

        lam = self.get_lambda(t_tensor, T, L).view(B, 1, 1)
        eps_final = (1 - lam) * noise_i + lam * eps  # lambda越大，白噪声越多

        return eps_final


class FreqDiffusion(nn.Module):
    def __init__(self, args, ):
        super(FreqDiffusion, self).__init__()
        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])

        self.noise_schedule = args.noise_schedule
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])

        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name,
                                                              self.num_timesteps)  ## lossaware (schedule_sample)
        self.timestep_map = self.time_map()
        self.rescale_timesteps = args.rescale_timesteps
        self.original_num_steps = len(betas)

        self.use_freq_noise = getattr(args, 'use_freq_noise', True)

        self.shared_s1 = nn.Parameter(torch.logit(torch.tensor(0.25)))
        self.shared_s2 = nn.Parameter(torch.logit(torch.tensor(0.5)))
        self.xstart_model = Diffu_xstart(self.hidden_size, args)
        self.xstart_model.freq_filer.raw_s1 = nn.Parameter(self.shared_s1 + 0.08)
        self.xstart_model.freq_filer.raw_s2 = nn.Parameter(self.shared_s2 - 0.08)

        if self.use_freq_noise:
            self.freq_noise_fn = FreqNoiseGenerator(
                seq_len=getattr(args, 'max_len', 50),
                device=args.device,
                shared_s1=self.shared_s1,
                shared_s2=self.shared_s2,
                k=8.0,
                smooth_sched="cosine"
            )
        else:
            print("[INFO] Using Standard Gaussian White Noise.")

    def get_betas(self, noise_schedule, diffusion_steps):
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)  ## array, generate beta
        return betas

    def q_sample(self, x_start, t, noise=None, mask=None, use_freq_noise=True, freq_noise_fn=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            if use_freq_noise and freq_noise_fn is not None:
                noise = freq_noise_fn(t, x_start.shape)
            else:
                noise = th.randn_like(x_start)

        scaling = noise.std(dim=(1, 2), keepdim=True)
        scaling = torch.clamp(scaling, 0.8, 1.2).detach()

        assert noise.shape == x_start.shape
        x_t = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise / (scaling + 1e-6)  ## reparameter trick
        )  ## genetrate x_t based on x_0 (x_start) with reparameter trick

        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return th.where(mask == 0, x_start, x_t)  ## replace the output_target_seq embedding (x_0) as x_t

<<<<<<< HEAD
    # 前向扩散
    def q_sample_freq(self, x_start, t, noise=None):
        X0 = torch.fft.rfft(x_start, dim=1, norm='ortho')

        if noise is None:
            if self.use_freq_noise and self.freq_noise_fn is not None:
                noise_time = self.freq_noise_fn(t, x_start.shape)
                noise = torch.fft.rfft(noise_time, dim=1, norm='ortho')
            else:
                noise = torch.randn_like(X0.real) + 1j * torch.randn_like(X0.real)

        scaling = noise.std(dim=(1, 2), keepdim=True)
        scaling = torch.clamp(scaling, 0.8, 1.2).detach()

        X_t = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, X0.shape) * X0
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, X0.shape)
                * noise / (scaling + 1e-6)
        )

        x_t = torch.fft.irfft(X_t, n=x_start.size(1), dim=1, norm='ortho')
        # x_t = x_t / (x_t.std(dim=(1, 2), keepdim=True) + 1e-6)
        return x_t, X_t

=======
>>>>>>> parent of 40f394a (引入频域的扩散v1)
    def time_map(self):
        timestep_map = []
        for i in range(len(self.alphas_cumprod)):
            if i in self.use_timesteps:
                timestep_map.append(i)
        return timestep_map

    # def scale_t(self, ts):
    #     map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
    #     new_ts = map_tensor[ts]
    #     # print(new_ts)
    #     if self.rescale_timesteps:
    #         new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
    #     return new_ts

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def _predict_xstart_from_eps(self, x_t, t, eps):

        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  ## \mu_t
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq):
        model_output, _ = self.xstart_model(rep_item, x_t, self._scale_timesteps(t), mask_seq)

        if model_output.dim() == 2:
            model_output = model_output.unsqueeze(1).expand_as(x_t)
        elif model_output.shape[1:] != x_t.shape[1:]:
            L_x0 = model_output.shape[1] if model_output.dim() > 2 else 1
            L_xt = x_t.shape[1]
            if L_x0 < L_xt:
                model_output = model_output.expand(-1, L_xt, -1)
            else:
                model_output = model_output[:, :L_xt, :]

        x_0 = model_output  ##output predict
        # x_0 = self._predict_xstart_from_eps(x_t, t, model_output)  ## eps predict

        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)

        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t,
                                                    t=t)  ## x_start: candidante item embedding, x_t: inputseq_embedding + outseq_noise, output x_(t-1) distribution
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq)
        # noise = th.randn_like(noise_x_t)
        if getattr(self, 'use_freq_noise', True):
            noise = self.freq_noise_fn(t, noise_x_t.shape)
        else:
            noise = th.randn_like(noise_x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))  # no noise when t == 0
        sample_xt = model_mean + nonzero_mask * th.exp(
            0.5 * model_log_variance) * noise  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick
        return sample_xt

    def reverse_p_sample(self, item_rep, noise_x_t, mask_seq):
        # device = next(self.xstart_model.parameters()).device

        if hasattr(self, "xstart_model") and any(p.requires_grad for p in self.xstart_model.parameters()):
            device = next(self.xstart_model.parameters()).device
        else:
            device = noise_x_t.device

        if noise_x_t.dim() == 2:
            # [B, H] → [B, L, H]
            noise_x_t = noise_x_t.unsqueeze(1).expand_as(item_rep)
        elif noise_x_t.dim() == 3 and noise_x_t.shape[1] != item_rep.shape[1]:
            L_item = item_rep.shape[1]
            L_noise = noise_x_t.shape[1]
            if L_noise > L_item:
                noise_x_t = noise_x_t[:, :L_item, :]
            else:
                pad_len = L_item - L_noise
                pad = torch.zeros((noise_x_t.size(0), pad_len, noise_x_t.size(2)), device=device)
                noise_x_t = torch.cat([noise_x_t, pad], dim=1)

        assert noise_x_t.shape == item_rep.shape, \
            f"Shape mismatch in reverse_p_sample: item_rep={item_rep.shape}, noise_x_t={noise_x_t.shape}"

        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:  # from T to 0, reversion iteration
            t = th.tensor([i] * item_rep.shape[0], device=device)
            with th.no_grad():
                noise_x_t = self.p_sample(item_rep, noise_x_t, t, mask_seq)
        return noise_x_t

<<<<<<< HEAD
    # 频域反向扩散
    def reverse_p_sample_freq(self, item_rep, X_t, mask_seq):
        device = item_rep.device
        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            t = torch.tensor([i] * item_rep.shape[0], device=device)

            # 还原为时域，用时域的预测网络xstart_model去噪估计
            x_t = torch.fft.irfft(X_t, n=item_rep.size(1), dim=1, norm='ortho')

            model_mean, model_log_variance = self.p_mean_variance(item_rep, x_t, t, mask_seq)

            # 预测结果重新转回频域
            model_mean_freq = torch.fft.rfft(model_mean, dim=1, norm='ortho')

            model_log_variance = _extract_into_tensor(
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                t, X_t.shape
            )

            # 生成频域噪声
            if getattr(self, 'use_freq_noise', True):
                noise_time = self.freq_noise_fn(t, item_rep.shape)
                noise = torch.fft.rfft(noise_time, dim=1, norm='ortho')
            else:
                noise = torch.randn_like(X_t)

            nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(X_t.shape) - 1))))

            X_t = model_mean_freq + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

        # 维持频谱能量一致（防止高频能量塌缩）
        # energy_target = torch.mean(torch.abs(torch.fft.rfft(item_rep, dim=1, norm='ortho')) ** 2, dim=(1, 2),
        #                            keepdim=True)
        # energy_current = torch.mean(torch.abs(X_t) ** 2, dim=(1, 2), keepdim=True)
        # X_t = X_t * torch.sqrt(energy_target / (energy_current + 1e-8))

        x_recon = torch.fft.irfft(X_t, n=item_rep.size(1), dim=1, norm='ortho')
        return x_recon

=======
>>>>>>> parent of 40f394a (引入频域的扩散v1)
    def forward(self, item_rep, item_tag, mask_seq):

        t, weights = self.schedule_sampler.sample(item_rep.shape[0],
                                                  item_tag.device)  ## t is sampled from schedule_sampler

        # t = self.scale_t(t)
        if self.use_freq_noise:
            noise = self.freq_noise_fn(t, item_rep.shape)
        else:
            noise = th.randn_like(item_rep)

        # 标准前向扩散，返回加入噪声的序列表示
        x_t = self.q_sample(item_rep, t, noise=noise)
        rep_time_out, item_rep_out = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)

        # ============ 频域扩散 ============
        x_t_freq, X_t = self.q_sample_freq(item_rep, t)
        rep_freq_out, _ = self.xstart_model(item_rep, x_t_freq, self._scale_timesteps(t), mask_seq)

<<<<<<< Updated upstream
        # eps, item_rep_out = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)  ## eps predict
        # x_0 = self._predict_xstart_from_eps(x_t, t, eps)

        # x_0, item_rep_out = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)  ##output predict
        return rep_time_out, rep_freq_out, item_rep_out, weights, t
=======
        # x_0:最终预测,item_rep_out：整个序列的特征表征
        x_0, item_rep_out = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)  ##output predict
        return x_0, item_rep_out, weights, t
>>>>>>> Stashed changes
