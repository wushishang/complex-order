# utils.py
import random
from collections import OrderedDict
import copy

import torch
from torchtext.legacy import data
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os 
import re

from common.helper import print_stats
from config import Config
from my_common.my_helper import is_positive_int
from util.constants import SentenceOrdering


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
    
    def parse_label(self, label):

        return int(label.strip()[0])

    def get_pandas_df(self, filename):

        full_df = pd.read_csv(filename, header=None, sep="\t", names=["text", "label"],encoding='gbk', quoting=3).fillna('N')
        return full_df

    def load_data(self, train_folder, dataset):
        
        if dataset=='sst2_transformer' or dataset=='TREC_transformer':
            print('no n fold cross validation')
            train_file = train_folder + '/train.csv'
            val_file = train_folder + '/dev.csv'
            test_file = train_folder + '/test.csv'
        else:
            if self.config.process_data:
                print_stats("Processing data...")
                process(train_folder, self.config)
            else:
                print_stats("Processed data exist! Skip data processing.")
            train_folder += f"/{self.config.experiment_data.name}_{self.config.n_fold}_{self.config.shuffle_random_state}"
            train_file = train_folder + f'/train_{self.config.n_fold}_{self.config.shuffle_random_state}_{self.config.cv_fold}.csv'
            val_file = train_folder + f'/dev_{self.config.n_fold}_{self.config.shuffle_random_state}_{self.config.cv_fold}.csv'
            test_file = train_folder + f'/test_{self.config.n_fold}_{self.config.shuffle_random_state}_{self.config.cv_fold}.csv'

        NLP = spacy.load('en_core_web_sm')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len,
                          include_lengths=True, pad_first=self.config.training_ordering==SentenceOrdering.pad_first)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text",TEXT),("label",LABEL)]
        
        train_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)
        order_sentences(train_data, self.config, 'train')

        val_df = self.get_pandas_df(val_file)
        val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
        val_data = data.Dataset(val_examples, datafields)
        order_sentences(val_data, self.config, 'train')


        TEXT.build_vocab(train_data)
        self.vocab = TEXT.vocab

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)
        
        self.val_iterator= data.BucketIterator(
             val_data,
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        test_df = self.get_pandas_df(test_file)
        if self.config.testing_ordering == SentenceOrdering.pad_first:
            TEXT_test = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len,
                              include_lengths=True, pad_first=True)
            test_datafields = [("text", TEXT_test), ("label", LABEL)]
            train_examples = [data.Example.fromlist(i, test_datafields) for i in train_df.values.tolist()]
            train_data = data.Dataset(train_examples, test_datafields)
            TEXT_test.build_vocab(train_data)
        else:
            test_datafields = datafields
        test_examples = [data.Example.fromlist(i, test_datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, test_datafields)
        order_sentences(test_data, self.config)
        self.test_iterator= data.BucketIterator(
            test_data,
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)
        
        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} val examples".format(len(val_data)))
        print("Loaded {} test examples".format(len(test_data)))

def clean_str_sst(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def balanced_cv_split(num_data, targets=None, num_classes=2, n_splits=5, random_state=1, shuffle=True):
    """
    A general balanced cv split
    Use: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    See also: https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/

    :return: a split indices generator that yields train/valid indices once
    """
    assert isinstance(num_data, int) and isinstance(num_classes, int)
    assert num_data > 1 and num_classes > 1
    # assert num_data % num_classes == 0  # May relax later but fine for now
    if targets is not None:
        assert targets.ndim == 1 and targets.shape[0] == num_data
        assert num_classes == len(np.unique(targets))
    else:
        # If no target is provided, assume the targets to be [0, 1, ..., num_classes-1]
        # num_per_class = int(num_data / num_classes)
        # targets = torch.tensor([], dtype=torch.long)
        # for i in range(num_classes):
        #     targets = torch.cat((targets, torch.tensor([i]).repeat(num_per_class)), dim=0)
        raise NotImplementedError("Haven't implemented balanced cv split without targets.")

    X_idx = np.zeros(num_data)
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    print_stats(f"Splitting data into {skf.get_n_splits(X_idx, targets)} folds...")

    # skf.split returns a generator (that yields train/valid indices)
    cv_splits = skf.split(X_idx, targets)

    cv_indices = OrderedDict()
    for fold, (train, test) in enumerate(cv_splits):
        cv_indices[fold] = {'train': tuple(train), 'test': tuple(test)}
    for fold, indices in cv_indices.items():
        indices['dev'] = copy.deepcopy(cv_indices[(fold+1)%len(cv_indices)]['test'])
        indices['train'] = tuple(set(indices['train']) - set(indices['dev']))
    return cv_indices

def process(dataset, cfg):
    data_dir = "../data/" + dataset
    saved_path = data_dir + f"/{cfg.experiment_data.name}_{cfg.n_fold}_{cfg.shuffle_random_state}"
    root = os.path.join(data_dir, "rt-polaritydata")
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    datas=[]
    for polarity in  ("neg","pos"):
        filename = os.path.join(root,polarity)
        records=[]
        with open(filename,encoding="utf-8",errors="replace") as f:
            for i,line in enumerate(f):
                records.append({"text":clean_str(line).strip(),"label": 1 if polarity == "pos" else 2})
        datas.append(pd.DataFrame(records))
    df = pd.concat(datas)

    cv_indices = balanced_cv_split(len(df), targets=df['label'], n_splits=cfg.n_fold,
                                   random_state=cfg.shuffle_random_state, shuffle=cfg.shuffle)
    for fold, indices in cv_indices.items():
        train = df.iloc[list(indices['train'])]
        dev = df.iloc[list(indices['dev'])]
        test = df.iloc[list(indices['test'])]
        train_filename = os.path.join(saved_path, f"train_{cfg.n_fold}_{cfg.shuffle_random_state}_{fold}.csv")
        dev_filename = os.path.join(saved_path, f"dev_{cfg.n_fold}_{cfg.shuffle_random_state}_{fold}.csv")
        test_filename = os.path.join(saved_path, f"test_{cfg.n_fold}_{cfg.shuffle_random_state}_{fold}.csv")
        train[["text", "label"]].to_csv(train_filename, encoding="utf-8", sep="\t", index=False, header=None)
        dev[["text", "label"]].to_csv(dev_filename, encoding="utf-8", sep="\t", index=False, header=None)
        test[["text", "label"]].to_csv(test_filename, encoding="utf-8", sep="\t", index=False, header=None)

    print("processing into formated files over")

def evaluate_model(model, iterator, eval=True):
    if eval:
        model.eval()
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text[0].cuda()
        else:
            x = batch.text[0]
        y_pred = model.predict(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score

def get_pe_variance(pe, original_mode=False, max_len=None):
    assert isinstance(original_mode, bool)
    if not original_mode:
        pe_weight = pe.weight.detach()  # pe is torch.Embedding()
    else:
        pe_weight = pe.detach().squeeze(0)  # pe is torch.randn() with first dim unsqueezed

    assert isinstance(pe_weight, torch.Tensor)
    assert pe_weight.ndim == 2
    if original_mode:
        assert is_positive_int(max_len)
        pe_weight = pe_weight[:max_len]

    pe_var = torch.var(pe_weight, dim=0)  # Var across 1st coordinate of each PE vector, 2nd coordinate, etc
    pe_var = torch.sum(pe_var).item()  # Sum the variances: 1. interpretable, 2. valid if independent
    norm = torch.norm(pe_weight).item()

    return pe_var, norm

def order_sentences(dataset, cfg: Config, mode='test'):
    assert isinstance(dataset, data.dataset.Dataset) and isinstance(cfg, Config)
    if mode == 'test':
        assert hasattr(cfg, 'testing_ordering')
        ordering = cfg.testing_ordering
        shuffle_seed = cfg.testing_shuffle_seed
    else:
        assert mode == 'train' and hasattr(cfg, 'training_ordering')
        ordering = cfg.training_ordering
        shuffle_seed = cfg.training_shuffle_seed

    if ordering == SentenceOrdering.random:
        for sentence in dataset.examples:
            random.Random(shuffle_seed).shuffle(sentence.text)
    elif ordering == SentenceOrdering.sort_up:
        for sentence in dataset.examples:
            sentence.text.sort()
    elif ordering == SentenceOrdering.sort_down:
        for sentence in dataset.examples:
            sentence.text.sort(reverse=True)
    elif ordering == SentenceOrdering.reverse:
        for sentence in dataset.examples:
            sentence.text.reverse()
    else:
        assert ordering in (SentenceOrdering.id, SentenceOrdering.pad_first)