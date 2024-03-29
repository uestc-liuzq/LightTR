import enum
import numpy as np
import torch


def evalandprint(args, LTR, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter,
                 best_changed):
    # evaluation on training data
    for client_idx in range(args.n_clients):
        train_loss, train_acc = LTR.client_eval(
            client_idx, train_loaders)
        print(
            f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')

    # evaluation on valid data
    val_acc_list = [None] * args.n_clients
    for client_idx in range(args.n_clients):
        val_loss, val_acc = LTR.client_eval(
            client_idx, val_loaders)
        val_acc_list[client_idx] = val_acc
        print(
            f' Site-{client_idx:02d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    if np.mean(val_acc_list) > np.mean(best_acc):
        for client_idx in range(args.n_clients):
            best_acc[client_idx] = val_acc_list[client_idx]
            best_epoch = a_iter
        best_changed = True

    if best_changed:
        best_changed = False
        # test
        for client_idx in range(args.n_clients):
            _, test_acc = LTR.client_eval(
                client_idx, test_loaders)
            print(
                f' Test site-{client_idx:02d} | Epoch:{best_epoch} | Test Acc: {test_acc:.4f}')
            best_tacc[client_idx] = test_acc
        print(f' Saving the local and server checkpoint to {SAVE_PATH}')
        tosave = {'best_epoch': best_epoch, 'best_acc': best_acc, 'best_tacc': np.mean(np.array(best_tacc))}
        for i, tmodel in enumerate(LTR.client_model):
            tosave['client_model_' + str(i)] = tmodel.state_dict()
        tosave['server_model'] = LTR.server_model.state_dict()
        torch.save(tosave, SAVE_PATH + 'val-best-model.pt')

    return best_acc, best_tacc, best_changed


def metricandprint(args, LTR, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter,
                   best_changed):
    # evaluation on training data
    for client_idx in range(args.n_clients):
        train_loss, train_acc = LTR.client_eval(
            client_idx, train_loaders)
        print(
            f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')

    # evaluation on valid data
    val_acc_list = [None] * args.n_clients
    for client_idx in range(args.n_clients):
        val_loss, val_acc = LTR.client_eval(
            client_idx, val_loaders)
        val_acc_list[client_idx] = val_acc
        print(
            f' Site-{client_idx:02d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    if np.mean(val_acc_list) > np.mean(best_acc):
        for client_idx in range(args.n_clients):
            best_acc[client_idx] = val_acc_list[client_idx]
            best_epoch = a_iter
        best_changed = True

    t_recall = []
    t_precision = []
    t_mae = []
    t_mse = []
    if best_changed:
        best_changed = False
        # test
        for client_idx in range(args.n_clients):
            _, test_acc = LTR.client_eval(
                client_idx, test_loaders)
            recall, precision, mae, mse = LTR.metrics(client_idx, test_loaders)
            print(
                f' Test site-{client_idx:02d} | Epoch:{best_epoch} | Test Acc: {test_acc:.4f}')
            best_tacc[client_idx] = test_acc
            t_recall.append(recall)
            t_precision.append(precision)
            t_mae.append(mae)
            t_mse.append(mse)
        print(f' Saving the local and server checkpoint to {SAVE_PATH}')
        tosave = {'best_epoch': best_epoch, 'best_acc': best_acc, 'best_tacc': np.mean(np.array(best_tacc))}
        for i, tmodel in enumerate(LTR.client_model):
            tosave['client_model_' + str(i)] = tmodel.state_dict()
        tosave['server_model'] = LTR.server_model.state_dict()
        torch.save(tosave, SAVE_PATH + 'val-best-model.pt')

    return sum(t_recall)/len(t_recall), sum(t_precision)/len(t_precision), sum(t_mae)/len(t_mae), sum(t_mse)/len(t_mse)
