# Word dataset
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class WordDataset(Dataset):
    
    def __init__(self, path_to_csv, word_col, label_col, reduce=True, remove_list=[]):
        
        self.df = pd.read_csv(path_to_csv)
        self.df = self.df.loc[np.logical_not(self.df.iloc[:, label_col].isnull()), :]
        self.word_col = word_col
        self.label_col = label_col
        all_chars = []
        self.df.iloc[:, word_col].apply(lambda x: all_chars.extend(list(x)))
        self.all_chars =  list(np.unique(all_chars)) 
        self.all_labels = list(self.df.iloc[:, label_col].unique())
        
        if remove_list:
            for rm in remove_list:
                self.df = self.df.loc[self.df.iloc[:, word_col] != rm, :]
        
        if reduce:
            self.df.iloc[:, word_col] = self.df.iloc[:, word_col].str.strip()
            self.df = self.df.loc[self.df.iloc[:, word_col] != '',:]
            self.df.drop_duplicates(subset=[self.df.columns[word_col], self.df.columns[label_col]],
                                   inplace=True)
            self.df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        word = self.df.iloc[idx, self.word_col]
        label = self.df.iloc[idx, self.label_col]
        word_tensor = torch.tensor([self.all_chars.index(c) + 1 for c in word])
        label_tensor = torch.tensor([self.all_labels.index(label)])
        
        return {'word':word_tensor, 'label':label_tensor, 'len': len(word_tensor)}