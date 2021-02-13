import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.utils import _pair, _quadruple
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from sklearn.utils import shuffle

class ML_adv_detector():
    '''Adversarial examples detector based on ml model
    
    Args:
        network - your torch NN used to compute softmax distribution
        detector - ML model with sklearn API
    '''
    
    def __init__(self, network, detector):
        self.network = network
        self.detector = detector
        
    def fit(self, X_real, X_adversarial):
        '''   
        X_real - torch dataloader
        X_adversarial - torch dataloader
        '''
        assert type(X_real) == torch.utils.data.dataloader.DataLoader
        assert type(X_adversarial) == torch.utils.data.dataloader.DataLoader
        
        X_softmax = []
        
        for x, y in X_real:
            x = x.to(device)
            outputs = F.softmax(self.network(x)).detach().cpu().numpy()
            X_softmax.extend(outputs)
        

        for x, y in X_adversarial:
            x = x.to(device)
            outputs = F.softmax(self.network(x)).detach().cpu().numpy()
            X_softmax.extend(outputs)  
       
        X_softmax = np.array(X_softmax)
        
        y = np.concatenate((np.array([0] * len(X_real.dataset)),
                            np.array([1] * len(X_adversarial.dataset))))
        
        X_softmax, y = shuffle(X_softmax, y)
        
        self.detector.fit(X_softmax, y)                    
            
    def predict(self, X):
        '''
        return result of detection (1 - adversarial, 0 - natural)
        X - torch dataloader
        '''
        X_softmax = []
        
        for x, y in X:
            x = x.to(device)
            outputs = F.softmax(self.network(x)).detach().cpu().numpy()
            X_softmax.extend(outputs)
        
        
        X_softmax = np.array(X_softmax)
        return np.round(self.detector.predict(X_softmax))
    

class Median_pooling(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(Median_pooling, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
    
    
class Filter_detector():
    '''Detector of adversarial examples based on image processing filters
    
    Args:
        size: size of kernel, int
    '''
    
    def __init__(self, model, filt=Median_pooling, size=3):
        self.network = model
        self.size = size
        self.filt = filt(kernel_size=self.size, same=True)
        
    def predict(self, X_loader):
        """X_loader - torch dataloader"""
        
        assert type(X_loader) == torch.utils.data.dataloader.DataLoader
        
        pre_filtered_pred = []
        for x, y in X_loader:
            x = x.to(device)
            outputs = torch.argmax(self.network(x), axis=1).detach().cpu().numpy()
            pre_filtered_pred.extend(outputs)
        
        pre_filtered_pred = np.array(pre_filtered_pred)
        
        filtered_pred = []
        for x, y in X_loader:
            x_fil = self.filt(x).to(device)
            outputs = torch.argmax(self.network(x_fil), axis=1).detach().cpu().numpy()
            filtered_pred.extend(outputs)
        filtered_pred = np.array(filtered_pred)
        
        return pre_filtered_pred != filtered_pred
    
    
class Combined_det():
    '''Adversarial examples detector based on ml model
    
    Args:
        network - your torch NN used to compute softmax distribution
        detector - ML model with sklearn API
    '''
    
    def __init__(self, network, detector, filt=Median_pooling(kernel_size=3, same=True)):
        self.network = network
        self.detector = detector
        self.filt = filt
        
    def fit(self, X_real, X_adversarial):
        '''   
        X_real - torch dataloader
        X_adversarial - torch dataloader
        '''
        assert type(X_real) == torch.utils.data.dataloader.DataLoader
        assert type(X_adversarial) == torch.utils.data.dataloader.DataLoader
        
        X_softmax = []
        
        for x, y in X_real:
            x = x.to(device)
            outputs = F.softmax(self.network(x)).detach().cpu()
            argmax = torch.argmax(outputs, axis=1)
            x_fil = self.filt(x).to(device)
            argmax_filt = torch.argmax(self.network(x_fil), axis=1).detach().cpu()
            
            flags = (argmax != argmax_filt).unsqueeze(1)
            outputs = torch.cat([outputs, flags], axis=1).numpy()
            
            X_softmax.extend(outputs)
        

        for x, y in X_adversarial:
            x = x.to(device)
            outputs = F.softmax(self.network(x)).detach().cpu()
            argmax = torch.argmax(outputs, axis=1)
            x_fil = self.filt(x).to(device)
            argmax_filt = torch.argmax(self.network(x_fil), axis=1).detach().cpu()
            
            flags = (argmax != argmax_filt).unsqueeze(1)
            outputs = torch.cat([outputs, flags], axis=1).numpy()
            
            X_softmax.extend(outputs)  
       
        X_softmax = np.array(X_softmax)
        
        y = np.concatenate((np.array([0] * len(X_real.dataset)),
                            np.array([1] * len(X_adversarial.dataset))))
        
        X_softmax, y = shuffle(X_softmax, y)
        
        self.detector.fit(X_softmax, y)                    
            
    def predict(self, X):
        '''
        return result of detection (1 - adversarial, 0 - natural)
        X - torch dataloader
        '''
        X_softmax = []
        
        for x, y in X:
            x = x.to(device)
            outputs = F.softmax(self.network(x)).detach().cpu()
            argmax = torch.argmax(outputs, axis=1)
            x_fil = self.filt(x).to(device)
            argmax_filt = torch.argmax(self.network(x_fil), axis=1).detach().cpu()
            
            flags = (argmax != argmax_filt).unsqueeze(1)
            outputs = torch.cat([outputs, flags], axis=1).numpy()
            X_softmax.extend(outputs)
        
        
        X_softmax = np.array(X_softmax)
        return np.round(self.detector.predict(X_softmax))