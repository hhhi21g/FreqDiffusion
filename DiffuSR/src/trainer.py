import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import time
import pickle


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item() / label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k - 1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg / max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit / log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics


def LSHT_inference(model_joint, args, data_loader):
    device = args.device
    model_joint = model_joint.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference.")
        model_joint = nn.DataParallel(model_joint)

    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]

            scores_rec, rep_diffu, _, _, _, _ = model_joint(test_batch[0], test_batch[1], train_flag=False)
            # scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu)

            if isinstance(model_joint, nn.DataParallel):
                scores_rec_diffu = model_joint.module.diffu_rep_pre(rep_diffu)
            else:
                scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu)

            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1], [5, 10, 20])
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print(test_metrics_dict_mean)


def model_train(tra_data_loader, val_data_loader, test_data_loader, model_joint, args, logger):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model_joint = model_joint.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model_joint = nn.DataParallel(model_joint)
    optimizer = optimizers(model_joint, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0,
                         'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0,
                  'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0

    with open("../datasets/data/amazon_beauty/use_ifo.pkl", "rb") as f:
        user_freq_info = pickle.load(f)

    # frequency warmup
    freq_warmup = getattr(args, "freq_warmup", 20)

    for epoch_temp in range(0, epochs):
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model_joint.train()

        flag_update = 0
        # running_loss = 0.0

        scaler = torch.cuda.amp.GradScaler()

        for idx, batch in enumerate(tra_data_loader):
            batch = [x.to(device) for x in batch]
            seq, label = batch[0], batch[1]
            # batch = [seq, label] 或 batch = [seq, label, user_id]
            if len(batch) >= 3:
                user_ids = batch[2]
            else:
                # 如果 DataLoader 没提供 user_id，则使用样本序号作为 user_id
                user_ids = torch.arange(seq.size(0), device=device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():

                # ============ 前向 ================
                x0_rep, seq_pred, weights, t, _, _ = model_joint(seq, label)  # seq_pred (B, L, H)

                if epoch_temp < freq_warmup:
                    freq_loss = torch.tensor(0.0, device=device)
                else:
                    # ============ FFT 主体 =============
                    # mask padding，避免噪声污染
                    mask = (seq > 0).float().unsqueeze(-1)
                    seq_masked = seq_pred * mask

                    fft_pred = torch.fft.rfft(seq_masked, dim=1, norm="ortho")  # B, F, H
                    fft_pred_mag = torch.abs(fft_pred).mean(dim=-1)  # B, F

                    B, F_pred = fft_pred_mag.shape
                    batch_info = [user_freq_info[int(uid.item())] for uid in user_ids]

                    max_F = max(
                        max(len(info["freq_global"]), len(info["freq_short"]), len(info["freq_period"]))
                        for info in batch_info
                    )

                    # 统一 pad 到 [B, max_F]
                    freq_global = torch.zeros(B, max_F, device=device)
                    freq_short = torch.zeros(B, max_F, device=device)
                    freq_period = torch.zeros(B, max_F, device=device)
                    P_list = torch.full((B,), -1, dtype=torch.long, device=device)

                    # 真实频谱拷贝（各自用自己的长度 Lg / Ls / Lp）
                    for i, uid_tensor in enumerate(user_ids):
                        uid = int(uid_tensor.item())
                        info = user_freq_info[uid]

                        fg = torch.tensor(info["freq_global"], device=device, dtype=torch.float32)
                        fs = torch.tensor(info["freq_short"], device=device, dtype=torch.float32)
                        fp = torch.tensor(info["freq_period"], device=device, dtype=torch.float32)
                        P = info["P"]

                        Lg = fg.shape[0]
                        Ls = fs.shape[0]
                        Lp = fp.shape[0]

                        freq_global[i, :Lg] = fg
                        freq_short[i, :Ls] = fs
                        freq_period[i, :Lp] = fp

                        if isinstance(P, int) and 0 <= P < max_F:
                            P_list[i] = P

                    # 归一化真实频谱
                    freq_global = freq_global / (freq_global.sum(dim=1, keepdim=True) + 1e-6)
                    freq_short = freq_short / (freq_short.sum(dim=1, keepdim=True) + 1e-6)
                    freq_period = freq_period / (freq_period.sum(dim=1, keepdim=True) + 1e-6)

                    # FFT 预测频谱与真实频谱对齐（按公共最小长度裁剪）
                    F = min(F_pred, max_F)

                    pred_f = fft_pred_mag[:, :F]
                    # pred_f = pred_f / (pred_f.sum(dim=1, keepdim=True) + 1e-6)

                    freq_global = freq_global[:, :F]
                    freq_short = freq_short[:, :F]
                    freq_period = freq_period[:, :F]

                    freq_idx = torch.arange(F, device=device, dtype=torch.float32).unsqueeze(0)

                    # low (长期)
                    # low (长期)
                    F_low = max(1, int(F * 0.25))
                    low_mask = (freq_idx <= F_low).float()
                    low_mask = low_mask / (low_mask.sum() + 1e-9)
                    L_low = ((pred_f - freq_global) ** 2 * low_mask).sum(dim=1).mean()

                    # high (短期)
                    F_high = max(F_low + 1, int(F * 0.50))
                    high_mask = (freq_idx >= F_high).float()
                    high_mask = high_mask / (high_mask.sum() + 1e-9)
                    L_high = ((pred_f - freq_short) ** 2 * high_mask).sum(dim=1).mean()

                    # mid (周期)
                    sigma = max(1.0, F * 0.08)

                    # 只保留 0 <= P < F 的样本
                    P_mask = ((P_list >= 0) & (P_list < F)).float()
                    P_center = P_list.clamp(0, F - 1).float().unsqueeze(1)  # [B,1]

                    weight_mid = torch.exp(
                        -(freq_idx - P_center) ** 2 / (2 * sigma ** 2)
                    )
                    weight_mid = weight_mid * P_mask.unsqueeze(1)
                    weight_mid = weight_mid / (weight_mid.sum(dim=1, keepdim=True) + 1e-9)

                    L_mid = ((pred_f - freq_period) ** 2 * weight_mid).sum(dim=1).mean()

                    # frequency loss
                    freq_loss = args.w_low * L_low + args.w_mid * L_mid + args.w_high * L_high

                # 主任务 loss
                if isinstance(model_joint, nn.DataParallel):
                    loss_unweighted = model_joint.module.loss_diffu_ce(x0_rep, label)
                else:
                    loss_unweighted = model_joint.loss_diffu_ce(x0_rep, label)

                loss_main = loss_unweighted.mean() if weights is None else (loss_unweighted * weights.detach()).mean()

                # warmup
                loss_all = loss_main + (args.lambda_freq * freq_loss if epoch_temp >= freq_warmup else 0)

            # backward
            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()

            if idx % max(1, int(len(tra_data_loader) / 5)) == 0:
                print(f"[{idx}/{len(tra_data_loader)}] Loss: {loss_all.item():.4f}")
                logger.info(f"[{idx}/{len(tra_data_loader)}] Loss: {loss_all.item():.4f}")
            if freq_loss:
                if idx % max(1, int(len(tra_data_loader) / 5)) == 0:
                    print(f"[{idx}/{len(tra_data_loader)}] Loss: {freq_loss.item() * args.lambda_freq:.4f}")
                    logger.info(f"[{idx}/{len(tra_data_loader)}] Loss: {freq_loss.item() * args.lambda_freq:.4f}")

        lr_scheduler.step()

        # if isinstance(model_joint, torch.nn.DataParallel):
        #     logs = model_joint.module.diffu.freq_noise_fn.param_log
        # else:
        #     logs = model_joint.diffu.freq_noise_fn.param_log
        #
        # if len(logs) > 0:
        #     start_mean = sum([x["start"] for x in logs]) / len(logs)
        #     end_mean = sum([x["end"] for x in logs]) / len(logs)
        #     tau_mean = sum([x["tau"] for x in logs]) / len(logs)
        #
        #     with open("lambda_track.txt", "a") as f:
        #         f.write(
        #             f"Epoch {epoch_temp + 1}: "
        #             f"start={start_mean:.4f}, end={end_mean:.4f}, tau={tau_mean:.4f}\n"
        #         )
        #
        #     # 记得清空记录，防止重复累计
        #     if isinstance(model_joint, torch.nn.DataParallel):
        #         model_joint.module.diffu.freq_noise_fn.param_log = []
        #     else:
        #         model_joint.diffu.freq_noise_fn.param_log = []

        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model_joint.eval()
            with torch.no_grad():
                metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
                # metrics_dict_mean = {}
                for val_batch in val_data_loader:
                    val_batch = [x.to(device) for x in val_batch]
                    seq_input = val_batch[0]
                    labels = val_batch[1]

                    if args.model.lower() == 'sasrec':
                        model_to_use = model_joint.module if isinstance(model_joint, nn.DataParallel) else model_joint
                        scores_all = model_to_use.full_sort_predict(seq_input)

                    else:
                        model_to_use = model_joint.module if isinstance(model_joint, nn.DataParallel) else model_joint
                        emb = model_to_use.item_embeddings(seq_input)  # [B, L, d]
                        z_T = torch.randn_like(emb)
                        rep_diffu = model_to_use.reverse(emb, z_T, (seq_input > 0).float())

                        last_idx = (seq_input > 0).sum(dim=1).clamp(min=1) - 1
                        rep_last = rep_diffu[torch.arange(rep_diffu.size(0)), last_idx]  # 取最后非padding位置
                        scores_all = model_to_use.diffu_rep_pre(rep_last)

                    metrics = hrs_and_ndcgs_k(scores_all, labels, metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)

            for key_temp, values_temp in metrics_dict.items():
                values_mean = round(np.mean(values_temp) * 100, 4)
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update = 1
                    bad_count = 0
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp

            if flag_update == 0:
                bad_count += 1
            else:
                print(best_metrics_dict)
                print(best_epoch)
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
                best_model = copy.deepcopy(model_joint)
            if bad_count >= args.patience:
                break

    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    if args.eval_interval > epochs:
        best_model = copy.deepcopy(model_joint)

    top_100_item = []
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]

            seq_input = test_batch[0]
            labels = test_batch[1]

            if args.model.lower() == 'sasrec':
                model_to_use = model_joint.module if isinstance(model_joint, nn.DataParallel) else model_joint
                scores_all = model_to_use.full_sort_predict(seq_input)
            else:
                model_to_use = model_joint.module if isinstance(model_joint, nn.DataParallel) else model_joint
                emb = model_to_use.item_embeddings(seq_input)  # [B, L, d]
                z_T = torch.randn_like(emb)  # 保持相同形状
                rep_diffu = model_to_use.reverse(emb, z_T, (seq_input > 0).float())

                # 使用最后一个有效位置表征
                last_idx = (seq_input > 0).sum(dim=1).clamp(min=1) - 1
                rep_last = rep_diffu[torch.arange(rep_diffu.size(0)), last_idx]
                scores_all = model_to_use.diffu_rep_pre(rep_last)

            # rep_diffu = rep_diffu.to(device)
            #
            # scores_rec_diffu = model_to_use.diffu_rep_pre(rep_diffu)
            # print(f"DEBUG scores_rec_diffu.shape: {scores_rec_diffu.shape}")

            k_eval = min(100, scores_all.size(-1))
            _, indices = torch.topk(scores_all, k=k_eval)

            # top_100_item.append(indices)

            metrics = hrs_and_ndcgs_k(scores_all, test_batch[1], metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)

    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print('Test------------------------------------------------------')
    logger.info('Test------------------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info(test_metrics_dict_mean)
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    print(args)

    if args.diversity_measure:
        path_data = '../datasets/data/category/' + args.dataset + '/id_category_dict.pkl'
        with open(path_data, 'rb') as f:
            id_category_dict = pickle.load(f)
        id_top_100 = torch.cat(top_100_item, dim=0).tolist()
        category_list_100 = []
        for id_top_100_temp in id_top_100:
            category_temp_list = []
            for id_temp in id_top_100_temp:
                category_temp_list.append(id_category_dict[id_temp])
            category_list_100.append(category_temp_list)
        category_list_100.append(category_list_100)
        path_data_category = '../datasets/data/category/' + args.dataset + '/DiffuRec_top100_category.pkl'
        with open(path_data_category, 'wb') as f:
            pickle.dump(category_list_100, f)

    # ----- 保存最优模型 -----
    save_path = f"./best_model_{args.dataset}.pt"
    if isinstance(best_model, nn.DataParallel):
        torch.save(best_model.module.state_dict(), save_path)
    else:
        torch.save(best_model.state_dict(), save_path)
    print(f"✅ Best model saved to {save_path}")

    return best_model, test_metrics_dict_mean
