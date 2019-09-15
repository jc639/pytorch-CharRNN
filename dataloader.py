# Dataloader that pads batches
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch

class WordDataLoader(object):
    
    def __init__(self, ds, batch_size=(32, 32), validation_split=0.1,
                shuffle=True, seed=42, device='cpu'):
        assert isinstance(batch_size, tuple)
        assert isinstance(validation_split, float)
        assert isinstance(shuffle, bool)
        assert isinstance(seed, int)
        assert isinstance(device, str)
        
        
        self.ds = ds
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.seed = seed
        self.device = device
        
    def  __call__(self):
        
        dataset_size = len(self.ds)
        unique_words = self.ds.df.iloc[:, self.ds.word_col].value_counts().index[self.ds.df.iloc[:, self.ds.word_col].value_counts() == 1]
        word_set = set(unique_words)
        unique_bool = [True if x in word_set else False for x in self.ds.df.iloc[:, self.ds.word_col].values]
        unique_idx = self.ds.df.index[unique_bool].values
        split = int(np.floor(self.validation_split * dataset_size))
        
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(unique_idx)
        val_indices = unique_idx[:split]
        remainder_indices = unique_idx[split:]
        train_indices = self.ds.df.index[np.logical_not(unique_bool)].values
        train_indices = np.append(train_indices, self.ds.df.index[remainder_indices].values)
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
                
        # Dataloader
        train_loader = DataLoader(self.ds, batch_size=self.batch_size[0],
                                  sampler=train_sampler, collate_fn=self.collate_fn)
        validation_loader = DataLoader(self.ds, batch_size=self.batch_size[1],
                                       sampler=valid_sampler, collate_fn=self.collate_fn)
        
        return train_loader, validation_loader
    
    def pad_sequences(self, word_tensor, tensor_lengths):
        seq_tensor = torch.zeros((len(word_tensor), tensor_lengths.max())).long()
        for idx, (seq, seqlen) in enumerate(zip(word_tensor, tensor_lengths)):
            seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)
        return seq_tensor   
    
    def sort_batch(self, seq_tensor, labels, tensor_lengths):
        seqlen, perm_idx = tensor_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        labels = labels[perm_idx]
        return seq_tensor, labels, seqlen

    def collate_fn(self, batch):
        
        word_tensor, label_tensor, tensor_lengths = [b.get('word') for b in batch], \
        [b.get('label') for b in batch], [b.get('len') for b in batch]
        labels = torch.cat(label_tensor)
        tensor_lengths = torch.tensor(tensor_lengths, dtype=torch.long)
        
        seq_tensor = self.pad_sequences(word_tensor, tensor_lengths)
        seq_tensor, labels, tensor_lengths = self.sort_batch(seq_tensor,
                                                            labels, 
                                                             tensor_lengths)
        
        if self.device == 'cpu':
            dev = torch.device('cpu')
        else:
            dev = torch.device('cuda')
        return seq_tensor.to(dev), labels.to(dev), tensor_lengths.to(dev)