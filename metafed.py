# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from common.mbr import MBR
from tqdm import tqdm
import time
from models.models_attn_tandem import Encoder, DecoderMulti, Seq2SeqMulti
from models.multi_train import evaluate, init_weights, train
from models.model_utils import epoch_time


class metafed(torch.nn.Module):
    def __init__(self, args, rn, rn_dict):
        super(metafed, self).__init__()
        self.server_model, self.client_model, self.client_weight = modelsel(
            args, args.device)
        self.optimizers = [optim.AdamW(params=self.client_model[idx].parameters(
        ), lr=args.learning_rate) for idx in range(args.n_clients)]
        self.loss_fun = nn.CrossEntropyLoss()
        args.sort = ''
        for i in range(args.n_clients):
            args.sort += '%d-' % i
        args.sort = args.sort[:-1]
        self.args = args
        self.csort = [int(item) for item in args.sort.split('-')]
        self.device = args.device
        self.mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
        self.rn = rn
        self.rn_dict = rn_dict
        self.thes = 0.5

    def init_model_flag(self, train_loaders, val_loaders, model_save_path):
        self.flagl = []
        client_num = self.args.n_clients
        for _ in range(client_num):
            self.flagl.append(False)
        optimizers = [optim.AdamW(params=self.client_model[idx].parameters(
        ), lr=self.args.learning_rate) for idx in range(client_num)]
        for idx in range(client_num):
            client_idx = idx
            model, train_loader, optimizer, tmodel, val_loader = self.client_model[
                                                                     client_idx], train_loaders, optimizers[
                                                                     client_idx], None, val_loaders
            for _ in range(30):
                _, _ = self.fed_train(model, train_loader, optimizer, tmodel, self.args, self.flagl[client_idx],
                                      model_save_path)
            _, val_acc, _,_,_,_ = self.fed_test(model, val_loader)
            if val_acc > self.args.threshold:
                self.flagl[idx] = True

    def update_flag(self, val_loaders):
        for client_idx, model in enumerate(self.client_model):
            _, val_acc, _,_,_,_ = self.fed_test(
                model, val_loaders)
            if val_acc > self.args.threshold:
                self.flagl[client_idx] = True

    def client_train(self, c_idx, dataloader, round, model_save_path):
        client_idx = self.csort[c_idx]
        tidx = self.csort[c_idx - 1]
        model, train_loader, optimizer, tmodel = self.client_model[
                                                     client_idx], dataloader, self.optimizers[client_idx], \
                                                 self.client_model[tidx]
        if round == 0 and c_idx == 0:
            tmodel = None
        for _ in range(self.args.wk_iters):
            train_loss, train_acc = self.fed_train(model, dataloader, optimizer, tmodel, self.args,
                                                   self.flagl[client_idx], model_save_path)
        return train_loss, train_acc

    def personalization(self, c_idx, dataloader, val_loader, model_save_path):
        client_idx = self.csort[c_idx]
        model, train_loader, optimizer, tmodel = self.client_model[
                                                     client_idx], dataloader, self.optimizers[
                                                     client_idx], copy.deepcopy(self.client_model[self.csort[-1]])

        with torch.no_grad():
            _, v1a, _,_,_,_ = self.fed_test(model, val_loader)
            _, v2a, _,_,_,_= self.fed_test(tmodel, val_loader)

        if v2a <= v1a and v2a < self.thes:
            lam = 0
        else:
            lam = (10 ** (min(1, (v2a - v1a) * 5))) / 10 * self.args.lam

        for _ in range(self.args.wk_iters):
            train_loss, train_acc = self.fed_train(model, dataloader, optimizer, tmodel, self.args,
                                                   self.flagl[client_idx], model_save_path)
        return train_loss, train_acc

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc, recall, precision, mae, mse = self.fed_test(
            self.client_model[c_idx], dataloader)
        return train_loss, train_acc

    def client_metrics(self, c_idx, dataloader):
        train_loss, train_acc, recall, precision, mae, mse = self.fed_test(
            self.client_model[c_idx], dataloader)
        return recall, precision, mae, mse

    def fed_train(self, model, iterator, optimizer, tmodel, args, flag, model_save_path):
        model.train()
        if tmodel:
            tmodel.eval()
            if not flag:
                with torch.no_grad():
                    for key in tmodel.state_dict().keys():
                        if 'num_batches_tracked' in key:
                            pass
                        elif args.nosharebn and 'bn' in key:
                            pass
                        else:
                            model.state_dict()[key].data.copy_(
                                tmodel.state_dict()[key])
        # Set mode to train model
        norm_grid_poi_dict, norm_grid_rnfea_dict, online_features_dict = None, None, None
        rid_features_dict = None
        ls_train_loss, ls_train_id_acc1, ls_train_id_recall, ls_train_id_precision, \
        ls_train_rate_loss, ls_train_id_loss = [], [], [], [], [], []
        dict_train_loss = {}
        epoch_loss = []

        # get all parameters (model parameters + task dependent log variances)
        log_vars = [torch.zeros((1,), requires_grad=True, device=self.device)] * 2  # use for auto-tune multi-task param
        self.iterator = iterator
        for epoch in tqdm(range(self.args.n_epochs)):
            start_time = time.time()

            new_log_vars, train_loss, train_id_acc1, train_id_recall, train_id_precision, \
            train_rate_loss, train_id_loss = train(model, self.iterator, optimizer, log_vars,
                                                   online_features_dict, rid_features_dict, self.args)

            ls_train_id_acc1.append(train_id_acc1)
            ls_train_id_recall.append(train_id_recall)
            ls_train_id_precision.append(train_id_precision)
            ls_train_rate_loss.append(train_rate_loss)
            ls_train_id_loss.append(train_id_loss)

            dict_train_loss['train_ttl_loss'] = ls_train_loss
            dict_train_loss['train_id_acc1'] = ls_train_id_acc1
            dict_train_loss['train_id_recall'] = ls_train_id_recall
            dict_train_loss['train_id_precision'] = ls_train_id_precision
            dict_train_loss['train_rate_loss'] = ls_train_rate_loss
            dict_train_loss['train_id_loss'] = ls_train_id_loss

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if (epoch % self.args.log_step == 0) or (epoch == self.args.n_epochs - 1):
                print('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
                weights = [torch.exp(weight) ** 0.5 for weight in new_log_vars]
                print('log_vars:' + str(weights))
                print('\tTrain Loss:' + str(train_loss) +
                      '\tTrain RID Acc1:' + str(train_id_acc1) +
                      '\tTrain RID Recall:' + str(train_id_recall) +
                      '\tTrain RID Precision:' + str(train_id_precision) +
                      '\tTrain Rate Loss:' + str(train_rate_loss) +
                      '\tTrain RID Loss:' + str(train_id_loss))

                torch.save(model.state_dict(), model_save_path + 'train-mid-model.pt')
            epoch_loss.append(train_loss)
        return sum(epoch_loss) / len(epoch_loss), sum(ls_train_id_acc1) / len(ls_train_id_acc1)

    def fed_test(self, model, dataloader):
        norm_grid_poi_dict, norm_grid_rnfea_dict, online_features_dict = None, None, None
        rid_features_dict = None

        ls_valid_loss, ls_valid_id_acc1, ls_valid_id_recall, ls_valid_id_precision, \
        ls_valid_dis_mae_loss, ls_valid_dis_rmse_loss = [], [], [], [], [], []
        ls_valid_dis_rn_mae_loss, ls_valid_dis_rn_rmse_loss, ls_valid_rate_loss, ls_valid_id_loss = [], [], [], []

        dict_train_loss = {}
        dict_valid_loss = {}
        best_valid_loss = float('inf')  # compare id loss
        self.iterator = dataloader
        # get all parameters (model parameters + task dependent log variances)
        log_vars = [torch.zeros((1,), requires_grad=True, device=self.device)] * 2  # use for auto-tune multi-task param
        for epoch in tqdm(range(self.args.n_epochs)):
            start_time = time.time()
            valid_id_acc1, valid_id_recall, valid_id_precision, \
            valid_rate_loss, valid_id_loss, valid_dis_mae_loss, valid_dis_rmse_loss, valid_dis_rn_mae_loss, \
            valid_dis_rn_rmse_loss, = evaluate(model, self.iterator, self.rn, self.rn_dict,
                                               online_features_dict, rid_features_dict, self.args)
            # valid_id_acc1, valid_id_recall, valid_id_precision, \
            # valid_rate_loss, valid_id_loss = evaluate(model, self.iterator, self.rn, self.rn_dict,
            #                                           online_features_dict, rid_features_dict, self.args)

            ls_valid_id_acc1.append(valid_id_acc1)
            ls_valid_id_recall.append(valid_id_recall)
            ls_valid_id_precision.append(valid_id_precision)
            ls_valid_dis_mae_loss.append(valid_dis_mae_loss)
            ls_valid_dis_rmse_loss.append(valid_dis_rmse_loss)
            ls_valid_dis_rn_mae_loss.append(valid_dis_rn_mae_loss)
            ls_valid_dis_rn_rmse_loss.append(valid_dis_rn_rmse_loss)
            ls_valid_rate_loss.append(valid_rate_loss)
            ls_valid_id_loss.append(valid_id_loss)
            valid_loss = valid_rate_loss + valid_id_loss
            ls_valid_loss.append(valid_loss)
            dict_valid_loss['valid_ttl_loss'] = ls_valid_loss
            dict_valid_loss['valid_id_acc1'] = ls_valid_id_acc1
            dict_valid_loss['valid_id_recall'] = ls_valid_id_recall
            dict_valid_loss['valid_id_precision'] = ls_valid_id_precision
            dict_valid_loss['valid_rate_loss'] = ls_valid_rate_loss
            dict_valid_loss['valid_dis_mae_loss'] = ls_valid_dis_mae_loss
            dict_valid_loss['valid_dis_rmse_loss'] = ls_valid_dis_rmse_loss
            dict_valid_loss['valid_dis_rn_mae_loss'] = ls_valid_dis_rn_mae_loss
            dict_valid_loss['valid_dis_rn_rmse_loss'] = ls_valid_dis_rn_rmse_loss
            dict_valid_loss['valid_id_loss'] = ls_valid_id_loss

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

            if (epoch % 49 == 0):
                print('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
                print('\tValid Loss:' + str(valid_loss) +
                      '\tValid RID Acc1:' + str(valid_id_acc1) +
                      '\tValid RID Recall:' + str(valid_id_recall) +
                      '\tValid RID Precision:' + str(valid_id_precision) +
                      '\tValid Distance MAE Loss:' + str(valid_dis_mae_loss) +
                      '\tValid Distance RMSE Loss:' + str(valid_dis_rmse_loss) +
                      # '\tValid Distance RN MAE Loss:' + str(valid_dis_rn_mae_loss) +
                      # '\tValid Distance RN RMSE Loss:' + str(valid_dis_rn_rmse_loss) +
                      '\tValid Rate Loss:' + str(valid_rate_loss) +
                      '\tValid RID Loss:' + str(valid_id_loss))

        return sum(ls_valid_loss) / len(ls_valid_loss), sum(ls_valid_id_acc1) / len(ls_valid_id_acc1), \
               sum(ls_valid_id_recall) / len(ls_valid_id_recall), sum(ls_valid_id_precision) / len(
            ls_valid_id_precision), \
               sum(ls_valid_dis_mae_loss) / len(ls_valid_dis_mae_loss), sum(ls_valid_dis_rmse_loss) / len(
            ls_valid_dis_rmse_loss)


def modelsel(args, device):
    enc = Encoder(args)
    dec = DecoderMulti(args)
    server_model = Seq2SeqMulti(enc, dec, device).to(device)
    server_model.apply(init_weights)  # learn how to init weights
    if args.load_pretrained_flag:
        server_model.load_state_dict(torch.load(args.model_old_path + 'val-best-model.pt'))

    print('model' + str(server_model))

    client_weights = [1 / args.n_clients for _ in range(args.n_clients)]
    models = [copy.deepcopy(server_model).to(device)
              for _ in range(args.n_clients)]
    return server_model, models, client_weights
