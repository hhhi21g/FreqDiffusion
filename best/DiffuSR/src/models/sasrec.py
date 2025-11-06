import numpy as np
import torch
import torch.nn.functional as F


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, args):
        super(SASRec, self).__init__()

        # self.user_num = user_num
        # self.item_num = item_num
        self.item_num = args.item_num
        self.hidden_units = args.hidden_units
        self.dev = args.device
        self.norm_first = args.norm_first

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):  # TODO: fp64 and int64 as default in python, trim?
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.tensor(log_seqs, dtype=torch.long, device=self.dev)
        else:
            log_seqs = log_seqs.long().to(self.dev)

        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        # # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        batch_size, seq_len = log_seqs.size()
        poss = torch.arange(1, seq_len + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1)
        poss = poss * (log_seqs != 0)
        seqs += self.pos_emb(poss)

        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                          attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                          attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, sequence, labels):
        # sequence: [B, L], labels: [B, 1]，label范围应为[1..item_num]，0是padding
        log_feats = self.log2feats(sequence)  # [B, L, D]
        final_feat = log_feats[:, -1, :]  # [B, D]
        all_item_embs = self.item_emb.weight  # [item_num+1, D]，0号是padding

        scores = torch.matmul(final_feat, all_item_embs.t())  # [B, item_num+1]
        scores[:, 0] = -1e9

        loss = F.cross_entropy(scores, labels.squeeze(-1), ignore_index=0)

        diffu_rep = final_feat
        return scores, diffu_rep, None, None, None, None, loss

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(item_indices.long().to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)

    def full_sort_predict(self, log_seqs: torch.Tensor) -> torch.Tensor:
        """
        输入: log_seqs [B, T] (cuda)
        输出: scores   [B, item_num]，对所有物品的打分，用于评估 HR/NDCG
        """
        with torch.no_grad():
            log_feats = self.log2feats(log_seqs)  # [B, T, H]
            final_feat = log_feats[:, -1, :]  # [B, H]
            # weight: [item_num+1, H]，去掉 padding 0 的那一行，得到 [item_num, H]
            all_item_emb = self.item_emb.weight[1:]  # [N, H]
            scores = final_feat @ all_item_emb.T  # [B, N]
        return scores

    @torch.no_grad()
    def full_sort_predict(self, sequence):
        self.eval()
        log_feats = self.log2feats(sequence)  # [B, L, D]
        final_feat = log_feats[:, -1, :]  # [B, D]
        all_item_embs = self.item_emb.weight  # [N+1, D]
        scores = torch.matmul(final_feat, all_item_embs.t())  # [B, N+1]
        scores[:, 0] = -1e9
        return scores
