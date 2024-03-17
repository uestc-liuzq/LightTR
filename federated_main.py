import logging
import time
import numpy as np
import argparse
import torch
from models.datasets import split_data,data_provider
from models.model_utils import load_rn_dict, load_rid_freqs, get_rid_grid, get_poi_info, get_rn_info
from models.model_utils import get_online_info_dict, epoch_time, AttrDict, get_rid_rnfea_dict
from models.models_LTR import Encoder, DecoderMulti, Seq2SeqMulti
from utils.utils import save_json_data, create_dir, load_pkl_data
from common.mbr import MBR
from common.road_network import load_rn_shp
from metafed import metafed
from federated_util.evalandprint import evalandprint, metricandprint


if __name__ == '__main__':
    start_time = time.time()


    parser = argparse.ArgumentParser(description='Lightweight Traj Recovery')
    parser.add_argument('--module_type', type=str, default='simple', help='module type')
    parser.add_argument('--keep_ratio', type=float, default=0.25, help='keep ratio in float')
    parser.add_argument('--lambda1', type=int, default=10, help='weight for multi task rate')
    parser.add_argument('--hid_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--grid_size', type=int, default=50, help='grid size in int')
    parser.add_argument('--dis_prob_mask_flag', action='store_true', help='flag of using prob mask')
    parser.add_argument('--pro_features_flag', action='store_true', help='flag of using profile features')
    parser.add_argument('--tandem_fea_flag', action='store_true', help='flag of using tandem rid features')
    parser.add_argument('--no_attn_flag', default=False, help='flag of using attention')
    parser.add_argument('--load_pretrained_flag', action='store_true', help='flag of load pretrained model')
    parser.add_argument('--model_old_path', type=str, default='', help='old model path')
    parser.add_argument('--no_debug', action='store_false', help='flag of debug')
    parser.add_argument('--no_train_flag', action='store_false', help='flag of training')
    parser.add_argument('--test_flag', action='store_true', help='flag of testing')
    parser.add_argument('--n_clients', type=int, default=10, help='number of clients')
    parser.add_argument('--fraction', type=float, default=1.0, help= 'fractions of clients')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='threshold to use copy or distillation, hyperparmeter for metafed')
    parser.add_argument('--lam', type=float, default=1.0,
                        help='init lam, hyperparmeter for metafed')
    parser.add_argument('--nosharebn', action='store_true',
                        help='not share bn')
    parser.add_argument('--global_epochs',type=int,help="the global epochs for training")
    opts = parser.parse_args()

    debug = opts.no_debug
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = AttrDict()
    args_dict = {
        'module_type': opts.module_type,
        'debug': debug,
        'device': device,
        'n_clients': opts.n_clients,
        'lam': opts.lam,
        'nosharebn': opts.nosharebn,
        'threshold': opts.threshold,
        'wk_iters':1,
        # pre train
        'load_pretrained_flag': opts.load_pretrained_flag,
        'model_old_path': opts.model_old_path,
        'train_flag': opts.no_train_flag,
        'test_flag': opts.test_flag,
        'epochs': opts.epochs,

        # constranit
        'dis_prob_mask_flag': opts.dis_prob_mask_flag,
        'search_dist': 50,
        'beta': 15,

        # features
        'tandem_fea_flag': opts.tandem_fea_flag,
        'pro_features_flag': opts.pro_features_flag,

        # extra info module
        'rid_fea_dim': 8,
        'pro_input_dim': 30,  # 24[hour] + 5[waether] + 1[holiday]
        'pro_output_dim': 8,
        'poi_num': 5,
        'online_dim': 5 + 5,  # poi/roadnetwork features dim
        'poi_type': 'company,food,shopping,viewpoint,house',
        # MBR
        'min_lat': 39.8887,
        'min_lng': 116.2683,
        'max_lat': 39.9866,
        'max_lng': 116.4565,

        # input data params
        'keep_ratio': opts.keep_ratio,
        'grid_size': opts.grid_size,
        'time_span': 5,
        'win_size': 20,
        'ds_type': 'random',
        'split_flag': False,
        'shuffle': True,

        # model params
        'hid_dim': opts.hid_dim,
        'id_emb_dim': 256,
        'dropout': 0.5,
        'id_size': 49853,

        'lambda1': opts.lambda1,
        'n_epochs': opts.epochs,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'tf_ratio': 0.5,
        'clip': 1,
        'log_step': 1
    }
    args.update(args_dict)

    print('Preparing data...')
    if args.split_flag:
        traj_input_dir = "./data/raw_trajectory/"
        output_dir = "./data/model_data/"
        split_data(traj_input_dir, output_dir)

    extra_info_dir = "./data/map/extra_info/"
    rn_dir = "./data/map/road_network/"
    train_trajs_dir = "./data/model_data/train_data/"
    valid_trajs_dir = "./data/model_data/valid_data/"
    test_trajs_dir = "./data/model_data/test_data/"
    if args.tandem_fea_flag:
        fea_flag = True
    else:
        fea_flag = False

    if args.load_pretrained_flag:
        model_save_path = args.model_old_path
    else:
        model_save_path = './results/' + args.module_type + '_kr_' + str(args.keep_ratio) + '_debug_' + str(
            args.debug) + \
                          '_gs_' + str(args.grid_size) + '_lam_' + str(args.lambda1) + \
                          '_prob_' + str(args.dis_prob_mask_flag) + \
                          '_fea_' + str(fea_flag) + '_' + time.strftime("%Y%m%d_%H%M%S") + '/'
        create_dir(model_save_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=model_save_path + 'log.txt',
                        filemode='a')

    rn = load_rn_shp(rn_dir, is_directed=True)
    rn_dict = load_rn_dict(extra_info_dir, file_name='rn_dict.json')

    mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
    # grid_rn_dict, max_xid, max_yid = get_rid_grid(mbr, args.grid_size, rn_dict)
    # args_dict['max_xid'] = max_xid
    # args_dict['max_yid'] = max_yid
    # args.update(args_dict)
    print(args)
    logging.info(args_dict)
    norm_grid_poi_dict, norm_grid_rnfea_dict, online_features_dict = None, None, None
    rid_features_dict = None
    train_teacher, valid_teacher,_ = data_provider(args,train_trajs_dir,valid_trajs_dir,test_trajs_dir,
                                                                mbr,norm_grid_poi_dict,norm_grid_rnfea_dict,debug)

    LTR = metafed(args,rn,rn_dict)
    LTR.init_model_flag(train_loaders=train_teacher, val_loaders=valid_teacher, model_save_path=model_save_path)
    args.epochs = args.epochs-1
    print('Common knowledge accumulation stage')

    best_changed = False

    best_acc = [0] * args.n_clients
    best_tacc = [0] * args.n_clients
    start_iter = 0

    for iter in range(start_iter, args.global_epochs):
        print(f"============ Train round============", iter)
        train_iterator, valid_iterator, test_iterator = data_provider(args, train_trajs_dir, valid_trajs_dir, test_trajs_dir,
                                                          mbr, norm_grid_poi_dict, norm_grid_rnfea_dict, debug)
        args.n_clients *= args.fraction
        for client in range(args.n_clients):
            LTR.client_train(
                client, train_iterator, iter, model_save_path)
        LTR.update_flag(valid_iterator)
        best_acc, best_tacc, best_changed = evalandprint(
            args, LTR, train_iterator, valid_iterator, test_iterator, model_save_path, best_acc, best_tacc, iter, best_changed)

        print('Personalization stage')
        for client in range(args.n_clients):
            LTR.personalization(
                client, train_iterator, valid_iterator,model_save_path)
        recall, precison, mae, mse = metricandprint(
            args, LTR, train_iterator, valid_iterator, test_iterator, model_save_path, best_acc, best_tacc, iter, best_changed)
        print(f'Recall:{recall},Precision{precison},MAE{mae},MSE{mse}')


    # for epoch in tqdm(range(global_epochs)):
    #     local_weights, local_losses = [], []
    #     print(f'\n | Global Training Round : {epoch+1} |\n')
    #
    #     global_model.train()
    #     m = max(num_users, 1)
    #     idxs_users = np.random.choice(range(num_users), m, replace=False)
    #
    #     for idx in idxs_users:
    #         local_model = LocalUpdate(args=args, dataset=train_dataset, iterator=train_iterator)
    #         w, loss = local_model.update_weights（
    #             model=copy.deepcopy(global_model), global_round=epoch,model_save_path=model_save_path)
    #         local_weights.append(copy.deepcopy(w))
    #         local_losses.append(copy.deepcopy(loss))
    #
    #     # update global weights
    #     global_weights = average_weights(local_weights)
    #
    #     # update global weights
    #     global_model.load_state_dict(global_weights)
    #
    #     loss_avg = sum(local_losses) / len(local_losses)
    #     train_loss.append(loss_avg)
    #
    #     # Calculate avg training accuracy over all users at every epoch
    #     list_acc, list_loss = [], []
    #     global_model.eval()
    #     for c in range(num_users):
    #         local_model = LocalUpdate(args=args, dataset=valid_dataset, iterator=valid_iterator)
    #         acc, loss = local_model.inference(model=global_model,model_save_path=model_save_path)
    #         list_acc.append(acc)
    #         list_loss.append(loss)
    #     train_accuracy.append(sum(list_acc)/len(list_acc))
    #
    #     # print global training loss after every 'i' rounds
    #     if (epoch+1) % print_every == 0:
    #         print(f' \nAvg Training Stats after {epoch+1} global rounds:')
    #         print(f'Training Loss : {np.mean(np.array(train_loss))}')
    #         print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
    #
    # # Test inference after completion of training
    # test_model = LocalUpdate(args=args, dataset=test_dataset, iterator=test_iterator)
    # test_id_acc1, test_id_precision, test_id_recall, test_dis_mae_loss, test_dis_rmse_loss, test_dis_rn_mae_loss, \
    # test_dis_rn_rmse_loss, test_rate_loss, test_id_loss = test_model.test_inference(args, global_model,model_save_path=model_save_path)
    #
    # print(f' \n Results after {global_epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100*test_id_acc1))
    #
    # # Saving the objects train_loss and train_accuracy:
    #
    # print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))