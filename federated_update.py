import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import copy
import time
import pandas as pd
from tqdm import tqdm
from models.multi_train import evaluate, init_weights, train
from models.model_utils import get_online_info_dict, epoch_time, AttrDict, get_rid_rnfea_dict
from common.mbr import MBR

class LocalUpdate(object):
    def __init__(self, args, dataset,iterator):
        self.args = args
        self.device = 'cuda:0'
        # Default criterion set to NLL loss function
        self.extra_info_dir = "./data/map/extra_info/"
        rn_dir = "./data/map/road_network/"
        self.dataset = dataset
        self.iterator = iterator
        self.mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)

    def update_weights(self, model, global_round, model_save_path):
        # Set mode to train model
        norm_grid_poi_dict, norm_grid_rnfea_dict, online_features_dict = None, None, None
        rid_features_dict = None

        print('global round:',global_round)
        # Set optimizer for the local updates
        ls_train_loss, ls_train_id_acc1, ls_train_id_recall, ls_train_id_precision, \
        ls_train_rate_loss, ls_train_id_loss = [], [], [], [], [], []
        dict_train_loss = {}
        epoch_loss = []

        # get all parameters (model parameters + task dependent log variances)
        log_vars = [torch.zeros((1,), requires_grad=True, device=self.device)] * 2  # use for auto-tune multi-task param
        optimizer = optim.AdamW(model.parameters(), lr=self.args.learning_rate)
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
        return model.state_dict(), sum(epoch_loss)/len(epoch_loss)

    def inference(self, model,model_save_path):
        """ Returns the inference accuracy and loss.
        """

        norm_grid_poi_dict, norm_grid_rnfea_dict, online_features_dict = None, None, None
        rid_features_dict = None

        ls_valid_loss, ls_valid_id_acc1, ls_valid_id_recall, ls_valid_id_precision, \
        ls_valid_dis_mae_loss, ls_valid_dis_rmse_loss = [], [], [], [], [], []
        ls_valid_dis_rn_mae_loss, ls_valid_dis_rn_rmse_loss, ls_valid_rate_loss, ls_valid_id_loss = [], [], [], []

        dict_train_loss = {}
        dict_valid_loss = {}
        best_valid_loss = float('inf')  # compare id loss

        # get all parameters (model parameters + task dependent log variances)
        log_vars = [torch.zeros((1,), requires_grad=True, device=self.device)] * 2  # use for auto-tune multi-task param
        optimizer = optim.AdamW(model.parameters(), lr=self.args.learning_rate)
        for epoch in tqdm(range(self.args.n_epochs)):
            start_time = time.time()

            valid_id_acc1, valid_id_recall, valid_id_precision, \
            valid_rate_loss, valid_id_loss = evaluate(model, self.iterator,
                                                      online_features_dict, rid_features_dict,self.args)

            ls_valid_id_acc1.append(valid_id_acc1)
            ls_valid_id_recall.append(valid_id_recall)
            ls_valid_id_precision.append(valid_id_precision)
            # ls_valid_dis_mae_loss.append(valid_dis_mae_loss)
            # ls_valid_dis_rmse_loss.append(valid_dis_rmse_loss)
            # ls_valid_dis_rn_mae_loss.append(valid_dis_rn_mae_loss)
            # ls_valid_dis_rn_rmse_loss.append(valid_dis_rn_rmse_loss)
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
                torch.save(model.state_dict(), model_save_path + 'val-best-model.pt')

            if (epoch%9==0):
                print('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
                print('\tValid Loss:' + str(valid_loss) +
                             '\tValid RID Acc1:' + str(valid_id_acc1) +
                             '\tValid RID Recall:' + str(valid_id_recall) +
                             '\tValid RID Precision:' + str(valid_id_precision) +
                             # '\tValid Distance MAE Loss:' + str(valid_dis_mae_loss) +
                             # '\tValid Distance RMSE Loss:' + str(valid_dis_rmse_loss) +
                             # '\tValid Distance RN MAE Loss:' + str(valid_dis_rn_mae_loss) +
                             # '\tValid Distance RN RMSE Loss:' + str(valid_dis_rn_rmse_loss) +
                             '\tValid Rate Loss:' + str(valid_rate_loss) +
                             '\tValid RID Loss:' + str(valid_id_loss))
        return valid_id_precision, valid_loss

    def test_inference(self,args, model,model_save_path):
        """ Returns the test accuracy and loss.
        """
        norm_grid_poi_dict, norm_grid_rnfea_dict, online_features_dict = None, None, None
        rid_features_dict = None

        model.load_state_dict(torch.load(model_save_path + 'val-best-model.pt'))
        start_time = time.time()
        test_id_acc1, test_id_recall, test_id_precision, test_rate_loss, test_id_loss = evaluate(model, self.iterator,
                                                                                             self.rn_dict,self.rn,
                                                                                             online_features_dict,
                                                                                             rid_features_dict,
                                                                                             args)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print('Test Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
        print('\tTest RID Acc1:' + str(test_id_acc1) +
                     '\tTest RID Recall:' + str(test_id_recall) +
                     '\tTest RID Precision:' + str(test_id_precision) +
                     # '\tTest Distance MAE Loss:' + str(test_dis_mae_loss) +
                     # '\tTest Distance RMSE Loss:' + str(test_dis_rmse_loss) +
                     # '\tTest Distance RN MAE Loss:' + str(test_dis_rn_mae_loss) +
                     # '\tTest Distance RN RMSE Loss:' + str(test_dis_rn_rmse_loss) +
                     '\tTest Rate Loss:' + str(test_rate_loss) +
                     '\tTest RID Loss:' + str(test_id_loss))

        return test_id_acc1,test_id_precision,test_id_recall,test_rate_loss, test_id_loss

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


