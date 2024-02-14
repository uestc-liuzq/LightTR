#!/usr/bin/python3
# coding: utf-8
# @Time    : 2020/9/2 19:58

import torch
import pickle
import json
import random
import os
import numpy as np
from models.datasets import Dataset, collate_fn, split_data


def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_pkl_data(data, dir, file_name):
    create_dir(dir)
    pickle.dump(data, open(dir + file_name, 'wb'))


def load_pkl_data(dir, file_name):
    '''
    Args:
    -----
        path: path
        filename: file name
    Returns:
    --------
        data: loaded data
    '''
    file = open(dir+file_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def save_json_data(data, dir, file_name):
    create_dir(dir)
    with open(dir+file_name, 'w') as fp:
        json.dump(data, fp)


def load_json_data(dir, file_name):
    with open(dir+file_name, 'r') as fp:
        data = json.load(fp)
    return data


def data_provider(args,train_trajs_dir,valid_trajs_dir,test_trajs_dir,mbr,norm_grid_poi_dict,norm_grid_rnfea_dict,debug):
    train_dataset = Dataset(train_trajs_dir, mbr=mbr, norm_grid_poi_dict=norm_grid_poi_dict,
                            norm_grid_rnfea_dict=norm_grid_rnfea_dict,
                            parameters=args, debug=debug)
    valid_dataset = Dataset(valid_trajs_dir, mbr=mbr, norm_grid_poi_dict=norm_grid_poi_dict,
                            norm_grid_rnfea_dict=norm_grid_rnfea_dict,
                            parameters=args, debug=debug)
    test_dataset = Dataset(test_trajs_dir, mbr=mbr, norm_grid_poi_dict=norm_grid_poi_dict,
                           norm_grid_rnfea_dict=norm_grid_rnfea_dict,
                           parameters=args, debug=debug)
    print('Finish data preparing.')
    print('training dataset shape: ' + str(len(train_dataset)))
    print('validation dataset shape: ' + str(len(valid_dataset)))
    print('test dataset shape: ' + str(len(test_dataset)))

    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                 shuffle=args.shuffle, collate_fn=collate_fn,
                                                 num_workers=4, pin_memory=True)
    valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                 shuffle=args.shuffle, collate_fn=collate_fn,
                                                 num_workers=4, pin_memory=True)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                shuffle=args.shuffle, collate_fn=collate_fn,
                                                num_workers=4, pin_memory=True)

    return train_iterator, valid_iterator, test_iterator