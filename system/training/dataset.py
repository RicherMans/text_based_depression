# -*- coding: utf-8 -*-
# @Author: richman
# @Date:   2018-01-18 10:28:31
# @Last Modified by:   richman
# @Last Modified time: 2018-04-11
import kaldi_io
import numpy as np
import torch
from torch.utils import data
from imblearn.over_sampling import RandomOverSampler


class ListDataset(torch.utils.data.Dataset):
    """Dataset wrapping List.

    Each sample will be retrieved by indexing List along the first dimension.

    Arguments:
        *lists (List): List that have the same size of the first dimension.
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(a_list) for a_list in lists)
        self.lists = lists

    def __getitem__(self, index):
        return tuple(a_list[index] for a_list in self.lists)

    def __len__(self):
        return len(self.lists[0])


def pad(tensorlist, batch_first=True, padding_value=0.):
    # In case we have 3d tensor in each element, squeeze the first dim (usually 1)
    if len(tensorlist[0].shape) == 3:
        tensorlist = [ten.squeeze() for ten in tensorlist]
    # In case of len == 1 padding will throw an error
    if len(tensorlist) == 1:
        return torch.as_tensor(tensorlist)
    tensorlist = [torch.as_tensor(item) for item in tensorlist]
    return torch.nn.utils.rnn.pad_sequence(tensorlist,
                                           batch_first=batch_first,
                                           padding_value=padding_value)


def sequential_collate(batches):
    # sort length wise
    batches.sort(key=lambda x: len(x), reverse=True)
    features, targets = zip(*batches)
    return pad(features), torch.as_tensor(targets)


def create_dataloader(kaldi_string,
                      label_dict,
                      transform=None,
                      batch_size: int = 16,
                      num_workers: int = 1,
                      shuffle: bool = True):
    """create_dataloader

    :param kaldi_string: copy-feats input
    :param label_dict: Labels from kaldi_data to manual value
    :param transform: Transformation on the data (scaling)
    :param batch_size: Batchsize
    :type batch_size: int
    :param num_workers: Number of workers in parallel
    :type num_workers: int
    """
    def valid_feat(item):
        """valid_feat
        Checks if feature is in labels

        :param item: key value pair from read_mat_ark
        """
        return item[0] in label_dict

    features = []
    labels = []
    # Directly filter out all utterances without labels
    for idx, (k, feat) in enumerate(
            filter(valid_feat, kaldi_io.read_mat_ark(kaldi_string))):
        if transform:
            feat = transform(feat)
        features.append(feat)
        labels.append(label_dict[k])
    assert len(features) > 0, "No features were found, are the labels correct?"
    # Shuffling means that this is training dataset, so oversample
    if shuffle:
        random_oversampler = RandomOverSampler(random_state=0)
        # Assume that label is Score, Binary, we take the binary to oversample
        sample_index = 1 if len(labels[0]) == 2 else 0
        # Dummy X data, y is the binary label
        _, _ = random_oversampler.fit_resample(torch.ones(
            len(features), 1), [l[sample_index] for l in labels])
        # Get the indices for the sampled data
        indicies = random_oversampler.sample_indices_
        # reindex, new data is oversampled in the minority class
        features, labels = [features[id] for id in indicies
                            ], [labels[id] for id in indicies]
    dataset = ListDataset(features, labels)
    # Configure weights to reduce number of unseen utterances
    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           collate_fn=sequential_collate,
                           shuffle=shuffle)
