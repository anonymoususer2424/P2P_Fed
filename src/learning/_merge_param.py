from collections import OrderedDict
import torch
from abc import abstractmethod
import copy
import numpy as np

class FedAvg_modif:
    def __init__(self):
        """Initialize FedAvg instance."""
        self.agg_weights = None
        self.aggregate_fn = self._aggregate_pytorch
                
    def do(self,
           par,
           P,
           *,
           dev,  
           total: int = 0,
           pre=None,
           **kwargs):

        assert (par is not None)

        self.par = par

        #data length for weighted avg
        total =0
        total=sum([p[1] for p in P])
        rates=[p[1]/total for p in P]
        print("total n: ",total,"rates: ",rates)

        self.agg_weights = copy.deepcopy(par)
        for key in par.keys():
            self.agg_weights[key] = par[key].to(torch.device(dev)) * rates[0]
        
        for k in range(1,len(P)):

            self.aggregate_fn(P[k][2], rates[k], dev)
        del self.par
        ret = self.agg_weights
        del self.agg_weights
        
        # for FedBuff case, save average data lengh of previous client sample.
        if pre is None:    
            return ret
        elif all(item is None for item in pre):
            pre[0] = total / 3
            pre[1] = ret
            return ret, pre         
        else:
            pre[0] = total / 4
            pre[1] = ret
            return ret, pre
        
    def _aggregate_pytorch(self, par, rate, dev):
        
        for k in par.keys():
            
            tmp= par[k] *rate
            tmp = tmp.to(torch.device(dev))
            self.agg_weights[k] += tmp