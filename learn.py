import pickle
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

optimiser = optim.Adam
loss_func = nn.CrossEntropyLoss()
CyclicScheduler = optim.lr_scheduler.CyclicLR  

class Learner(object):
    
    def __init__(self, model, dataloader, all_chars, all_labels,
                 loss_func=loss_func, optimiser=optimiser, 
                 scheduler=CyclicScheduler):
        
        self.train_dl, self.valid_dl = dataloader()
        self.model = model
        self.loss_func = loss_func
        self.optimiser = optimiser 
        self.opt = None
        self.scheduler = scheduler
        self.sched = None
        self.best_acc = 0
        self.all_chars = all_chars
        self.all_labels = all_labels
        
    def fit_one_cycle(self, epochs, max_lr, base_lr=None, save_best_weights=False,
                      base_moms=0.8, max_moms=0.9, wd=1e-2):
        if base_lr is None:
            base_lr = max_lr / 10

        total_batches = epochs * len(self.train_dl)
        up_size = np.floor(total_batches * 0.25)
        down_size = np.floor(total_batches*0.95 - up_size)
        
        self.opt = self.optimiser(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.opt.defaults['momentum'] = 0.9
        self.opt.param_groups[0]['momentum'] = 0.9
        self.opt.param_groups[0]['weight_decay'] = wd
        
        self.sched = self.scheduler(self.opt, max_lr=max_lr, base_lr=base_lr, base_momentum=base_moms,
                       max_momentum=max_moms, step_size_up=up_size, step_size_down=down_size)
        self.opt.param_groups[0]['betas'] = (self.opt.param_groups[0]['momentum'], self.opt.param_groups[0]['betas'][1]) 
        
        self._fit(epochs=epochs, cyclic=True)
    
    def fit(self, epochs, lr=1e-3, wd=1e-2, betas=(0.9, 0.999)):
        
        self.opt = self.optimiser(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                  weight_decay=wd, betas=betas)
        self._fit(epochs=epochs, cyclic=False)
        
    def _fit(self, epochs, cyclic=False):
        
        len_train = len(self.train_dl)
        # fit
        for i in range(1, epochs + 1):
            self.model.train()
            batch_n = 1
            train_loss = 0
            loss = 0
            for xb, yb, lens in self.train_dl:
                print('epoch {}: batch {} out of {} | loss {}'.format(i, batch_n, len_train, loss), end='\r',
                      flush=True)
                self.opt.zero_grad()
                out = self.model(xb, lens)
                loss = self.loss_func(out, yb)
                
                with torch.no_grad():
                    train_loss += loss
                
                loss.backward()
                self.opt.step()
                if cyclic:
                    if self.sched.last_epoch < self.sched.total_size:
                        self.sched.step()
                        self.opt.param_groups[0]['betas'] = (self.opt.param_groups[0]['momentum'], self.opt.param_groups[0]['betas'][1])
                    
                batch_n += 1
                
                
            self.model.eval()
            with torch.no_grad():
                
                valid_loss = 0
                acc = 0 
                for xb, yb, lens in self.valid_dl:
                    out = self.model(xb, lens)
                    valid_loss += self.loss_func(out, yb)
                    acc += (out.softmax(1).argmax(1) == yb).sum().item()
                    
                acc = acc / len(self.valid_dl.batch_sampler.sampler.indices)
                
            print('epoch {}: train loss {} | valid loss {} | acc {}'.format(i, train_loss / len(self.train_dl),
                  valid_loss / len(self.valid_dl), acc),
                  end='\n')
            
            if acc > self.best_acc:
                self.save()
                self.best_acc = acc
        
    def find_lr(self, start_lr, end_lr, wd=1e-2, momentum=0.9, num_interval=200, plot=True):
        
        # store the state dict at start so we can restore it
        sd = self.model.state_dict()
        
        # number of mini-batches
        if num_interval < len(self.train_dl):
            num = num_interval
        else:
            num = len(self.train_dl) - 1
        multi = (end_lr / start_lr) ** (1/num)
        lr = start_lr
        self.opt = self.optimiser(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.opt.param_groups[0]['lr'] = lr
        self.opt.param_groups[0]['weight_decay'] = wd
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        lrs = []
        for xb, yb, lens in self.train_dl:
            batch_num += 1
            print('batch {}'.format(batch_num), end='\r',
                      flush=True)
            self.model.train()
            out = self.model(xb, lens)
            loss = self.loss_func(out, yb)
            avg_loss = momentum * avg_loss + (1-momentum) * loss.data.item()
            smoothed_loss = avg_loss / (1-momentum**batch_num)
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                self.model.load_state_dict(sd)
                if plot:
                    plt.semilogx(lrs, losses)
                    plt.show()
                return lrs, losses
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            losses.append(smoothed_loss)
            lrs.append(lr)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            lr *= multi
            self.opt.param_groups[0]['lr'] = lr
        
        self.model.load_state_dict(sd)
        if plot:
            plt.semilogx(lr, losses)
            plt.show()
        return lrs, losses
    
    def save(self, f='model.pth'):
        
        try:
            torch.save(self.model.state_dict(), f=f)
            with open('character_label.pkl', 'wb') as file:
                pickle.dump({'all_chars':self.all_chars,
                            'all_labels':self.all_labels}, file)
        except OSError as e:
            print(e)
        
    def load(self, f='model.pth'):
        
        try:
            self.model.load_state_dict(torch.load(f))
            self.model.eval()
            with open('character_label.pkl', 'rb') as file:
                char_dict = pickle.load(file)
                self.all_chars = char_dict['all_chars']
                self.all_labels = char_dict['all_labels']
        except OSError as e:
            print(e)
               
    def predict(self, line):
        print(f'\n > {line}')
        line = line.upper()
        line_tensor = torch.tensor([[self.all_chars.index(l) + 1 if l in self.all_chars else self.all_chars.index(' ') + 1 for l in line]])
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(line_tensor.cuda(), torch.tensor([len(line)], dtype=torch.long))
            val, ind  = out.softmax(1).topk(3, dim=1)
            
            for i in range(3):
                value = val[0][i].item()
                cat_idx = ind[0][i].item()
                print('({:.2f}) {}'.format(value, self.all_labels[cat_idx]))
    