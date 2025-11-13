import torch.nn as nn
import torch
import math
from .freqdiffusion import FreqDiffusion
import torch.nn.functional as F
import copy
import numpy as np
from .step_sample import LossAwareSampler
import torch as th


def freqband_consistency_loss(X_in, X_pred, s1, s2, k=20.0, scale=4.0, use_mid=True, epoch=None):
    """中/高频一致性（teacher停梯度、同分母归一、通道均值+频率求和+有效频点归一）"""
    eps = 1e-6
    B, F, H = X_in.shape
    u = torch.linspace(0, 1, F, device=X_in.device)[None, :, None]

    # 中/高频 soft mask
    high_mask = torch.sigmoid(k * (u - s2))  # [1,F,1]
    band_mask = torch.sigmoid(k * (u - s1)) * torch.sigmoid(k * (s2 - u))  # [1,F,1]

    # if not use_mid:
    # print("no mid")
    # band_mask = torch.zeros_like(band_mask)
    band_mask = band_mask * use_mid

    E = X_in.abs().mean(dim=(1, 2), keepdim=True) + eps
    energy_weight = (X_in.abs() / E).pow(0.5).detach()

    # —— 同一分母归一化（以 X_in 为标尺），teacher 停梯度 ——
    denom = X_in.abs().mean(dim=(1, 2), keepdim=True).detach() + eps
    X_in_n = X_in.detach() / denom
    X_pred_n = X_pred / denom

    # log1p 幅谱（高频小能量更稳）
    S_in = torch.log1p(X_in_n.abs())
    S_pred = torch.log1p(X_pred_n.abs())

    # 差异：通道均值 → 频率求和
    diff2 = (S_pred - S_in).pow(2) * energy_weight  # [B,F,H]
    diff_mid = (diff2 * band_mask).mean(dim=2).sum(dim=1)  # [B]
    diff_high = (diff2 * high_mask).mean(dim=2).sum(dim=1)  # [B]

    # 按有效频点归一（防止 s1/s2 面积不同）
    F_eff_mid = band_mask.sum(dim=1).squeeze(-1) + eps
    F_eff_high = high_mask.sum(dim=1).squeeze(-1) + eps
    diff_mid = diff_mid / F_eff_mid
    diff_high = diff_high / F_eff_high

    if epoch is not None:
        mix_ratio = torch.clamp(torch.tensor(epoch / 100.0, device=X_in.device), 0, 1).item()
    else:
        mix_ratio = 0.5
    w_high = 0.7 - 0.2 * mix_ratio
    w_mid = 0.3 + 0.2 * mix_ratio

    L = scale * (w_high * diff_high + w_mid * diff_mid).mean()
    return L


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


class Att_Diffuse_model(nn.Module):
    def __init__(self, diffu, args):
        super(Att_Diffuse_model, self).__init__()
        self.args = args
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num + 1
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = diffu
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
        self.momentum = 0.99
        self.register_buffer("ema_inited", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer("ema_ctx", torch.zeros(1, 1, 1))

    def diffu_pre(self, item_rep, tag_emb, mask_seq):
        seq_rep_diffu, item_rep_out, weights, t = self.diffu(item_rep, tag_emb, mask_seq)
        return seq_rep_diffu, item_rep_out, weights, t

    def reverse(self, item_rep, noise_x_t, mask_seq):
        reverse_pre = self.diffu.reverse_p_sample(item_rep, noise_x_t, mask_seq)
        return reverse_pre

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_diffu(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        scores_pos = scores.gather(1, labels)  ## labels: b x 1
        scores_neg_mean = (torch.sum(scores, dim=-1).unsqueeze(-1) - scores_pos) / (scores.shape[1] - 1)

        loss = torch.min(-torch.log(torch.mean(torch.sigmoid((scores_pos - scores_neg_mean).squeeze(-1)))),
                         torch.tensor(1e8))

        # if isinstance(self.diffu.schedule_sampler, LossAwareSampler):
        #     self.diffu.schedule_sampler.update_with_all_losses(t, loss.detach())
        # loss = (loss * weights).mean()
        return loss

    def loss_diffu_ce(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        """
        return self.loss_ce(scores, labels.squeeze(-1))

    def diffu_rep_pre(self, rep_diffu):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return scores

    def loss_rmse(self, rep_diffu, labels):
        rep_gt = self.item_embeddings(labels).squeeze(1)
        return torch.sqrt(self.loss_mse(rep_gt, rep_diffu))

    def routing_rep_pre(self, rep_diffu):
        item_norm = (self.item_embeddings.weight ** 2).sum(-1).view(-1, 1)  ## N x 1
        rep_norm = (rep_diffu ** 2).sum(-1).view(-1, 1)  ## B x 1
        sim = torch.matmul(rep_diffu, self.item_embeddings.weight.t())  ## B x N
        dist = rep_norm + item_norm.transpose(0, 1) - 2.0 * sim
        dist = torch.clamp(dist, 0.0, np.inf)

        return -dist

    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep / seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep / item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep / seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1) / torch.sum(mask_seq, dim=-1))

    def forward(self, sequence, tag, train_flag=True):

        seq_length = sequence.size(1)

        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)  ## dropout first than layernorm

        # item_embeddings = item_embeddings + position_embeddings

        item_embeddings = self.LayerNorm(item_embeddings)

        mask_seq = (sequence > 0).float()

        if train_flag:
            tag_emb = self.item_embeddings(tag.squeeze(-1))  ## B x H
            rep_diffu, rep_item, weights, t = self.diffu_pre(item_embeddings, tag_emb, mask_seq)

            with torch.no_grad():
                if self.ema_ctx.numel() == 1 or self.ema_ctx.shape != rep_item.shape:
                    self.ema_ctx = rep_item.detach().clone().to(rep_item.device)
                    self.ema_inited.fill_(1)
                else:
                    with torch.no_grad():
                        cur_epoch = getattr(self.args, "epoch", 0)
                        if cur_epoch < 100:
                            cur_mom = 0.97
                        else:
                            cur_mom = 0.99
                        self.ema_ctx.mul_(cur_mom).add_(rep_item, alpha=1 - cur_mom)

                    # teacher_ctx = self.ema_ctx.detach()

            teacher_ctx = self.ema_ctx.detach()  # 停梯度的 teacher

            eps = 1e-6
            # x_in = item_embeddings * mask_seq.unsqueeze(-1)
            x_in = teacher_ctx * mask_seq.unsqueeze(-1)
            x_pred = rep_item * mask_seq.unsqueeze(-1)

            k = min(16, x_in.size(1))
            x_in = x_in[:, -k:, :]
            x_pred = x_pred[:, -k:, :]
            mk = mask_seq[:, -k:]

            hann = torch.hann_window(k, periodic=False, device=x_in.device).view(1, k, 1)
            x_in = x_in * mk.unsqueeze(-1) * hann
            x_pred = x_pred * mk.unsqueeze(-1) * hann

            x_in = torch.fft.rfft(x_in, dim=1, norm='ortho')
            x_pred = torch.fft.rfft(x_pred, dim=1, norm='ortho')

            s1 = torch.sigmoid(self.diffu.shared_s1)
            s2 = torch.sigmoid(self.diffu.shared_s2)

            base_scale = getattr(self.args, "consist_scale", 2.5)
            cur_epoch = getattr(self.args, "epoch", 0)

            # scale 调度：40→120 线性从 2.5 → 4.0
            if cur_epoch < 40:
                scale_eff = 0.0
            elif cur_epoch < 120:
                scale_eff = base_scale + (4.0 - base_scale) * ((cur_epoch - 40) / 80.0)
            else:
                scale_eff = 4.0

            if getattr(self.args, "epoch", 0) >= 30:  # 前40轮不加入L_consist
                L_consist = freqband_consistency_loss(
                    x_in, x_pred, s1, s2,
                    k=20.0,
                    scale=scale_eff,
                    use_mid=getattr(self.args, "use_mid", True),
                    epoch=getattr(self.args, "epoch", 0)
                )
            else:
                L_consist = None

            item_rep_dis = None
            seq_rep_dis = L_consist
        else:
            # noise_x_t = th.randn_like(tag_emb)
            noise_x_t = th.randn_like(item_embeddings[:, -1, :])
            rep_diffu = self.reverse(item_embeddings, noise_x_t, mask_seq)
            if rep_diffu.dim() == 3:
                rep_diffu = rep_diffu.mean(dim=1)
            weights, t, item_rep_dis, seq_rep_dis = None, None, None, None

        scores = None
        return scores, rep_diffu, weights, t, item_rep_dis, seq_rep_dis


def create_model_diffu(args):
    diffu_pre = FreqDiffusion(args)
    return diffu_pre
