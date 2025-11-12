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

    for epoch_temp in range(0, epochs):
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model_joint.train()

        flag_update = 0
        # running_loss = 0.0

        # ======== 动态一致性与EMA策略调度 ========
        total_epochs = epochs

        # ---- 动态 EMA momentum ----
        if epoch_temp < 20:
            model_joint.momentum = 0.97
        elif epoch_temp < 80:
            model_joint.momentum = 0.985
        elif epoch_temp < 150:
            model_joint.momentum = 0.99
        else:
            model_joint.momentum = 0.995

        # ---- 动态一致性 λ (curriculum) ----
        if epoch_temp < 20:
            lam = 0.1 + 0.4 * (epoch_temp / 20)  # 0.1→0.5
        elif epoch_temp < 80:
            lam = 0.5 + 0.5 * ((epoch_temp - 20) / 60)  # 0.5→1.0
        elif epoch_temp < 150:
            lam = 1.0  # plateau
        elif epoch_temp < 220:
            lam = 1.0 + 0.5 * ((epoch_temp - 180) / 70)  # 1.0→1.5
        else:
            lam = 1.5 - 1.0 * ((epoch_temp - 250) / 50)  # 1.5→0.5
        args.consist_lambda = lam

        # ---- 动态频带控制 ----
        # 前5轮只用高频，之后用中+高频
        if epoch_temp < 5:
            args.use_mid = False
        elif epoch_temp < 250:
            args.use_mid = True
        else:
            args.use_mid = False  # fine-tune 阶段再去掉中频

        scaler = torch.cuda.amp.GradScaler()

        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]
            seq, label = train_batch[0], train_batch[1]  # ✅ 提前统一解包
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                if args.model.lower() == 'sasrec':
                    scores, diffu_rep, _, _, _, _, loss_all = model_joint(seq, label)
                    loss_all = loss_all.mean()
                else:
                    _, diffu_rep, weights, t, _, L_consist = model_joint(seq, label)
                    if isinstance(model_joint, nn.DataParallel):
                        loss_unweighted = model_joint.module.loss_diffu_ce(diffu_rep, label)
                    else:
                        loss_unweighted = model_joint.loss_diffu_ce(diffu_rep, label)
                    if weights is not None:
                        loss_all = (loss_unweighted * weights.detach()).mean()
                    else:
                        loss_all = loss_unweighted.mean()

                    # lam = getattr(args,"consist_lambda",1)
                    if L_consist is None:
                        loss_all = loss_all
                    else:
                        loss_all = loss_all + args.consist_lambda * L_consist

            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss_all.backward()
            #
            # optimizer.step()
            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
                logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
                print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), L_consist.item()))
                logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), L_consist.item()))
        print("loss in epoch {}: {}".format(epoch_temp, loss_all.item()))
        lr_scheduler.step()

        if isinstance(model_joint, torch.nn.DataParallel):
            logs = model_joint.module.diffu.freq_noise_fn.param_log
        else:
            logs = model_joint.diffu.freq_noise_fn.param_log

        if len(logs) > 0:
            start_mean = sum([x["start"] for x in logs]) / len(logs)
            end_mean = sum([x["end"] for x in logs]) / len(logs)
            tau_mean = sum([x["tau"] for x in logs]) / len(logs)

            with open("lambda_track.txt", "a") as f:
                f.write(
                    f"Epoch {epoch_temp + 1}: "
                    f"start={start_mean:.4f}, end={end_mean:.4f}, tau={tau_mean:.4f}\n"
                )

            # 记得清空记录，防止重复累计
            if isinstance(model_joint, torch.nn.DataParallel):
                model_joint.module.diffu.freq_noise_fn.param_log = []
            else:
                model_joint.diffu.freq_noise_fn.param_log = []

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
