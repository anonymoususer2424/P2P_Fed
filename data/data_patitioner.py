import os.path
import pickle

import numpy as np
import torchvision
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms   
import argparse

class custom_dataset(VisionDataset):

    def __init__(
        self,
        root:str,
        transform=None,
        download=True,
        split_number=10,
        iid=False,
        dataset_name="cifar10",
        unbalanced_sgm=0,
        rule_arg=0,
        sharding=0
    ) -> None:
        super(custom_dataset, self).__init__(root=root, transform=transform)
        self.dataset_name = dataset_name
        self.train = True
        self.root =root
        self.transform = transforms.Compose([transforms.ToTensor()]) 
        self.download = download
        self.split_number = split_number
        self.iid = iid
        self.data = []
        self.targets = []
        self.unbalanced_sgm = unbalanced_sgm
        self.rule_arg = rule_arg
        self.sharding = sharding
        self.get_data()
        print("Param info : unbal ",unbalanced_sgm,"rule_arg", rule_arg, "sharding", sharding)
        
    def get_data(self):
        
        # get original dataset
        
        if self.dataset_name == "cifar10":
            dataset = torchvision.datasets.CIFAR10(root=self.root, train=self.train, download=self.download, transform = transforms.Compose([transforms.ToTensor(),]))
        elif self.dataset_name == "cifar100":
            dataset = torchvision.datasets.CIFAR100(root=self.root, train=self.train, download=self.download, transform = transforms.Compose([transforms.ToTensor(),]))
        elif self.dataset_name == "mnist":
            dataset = torchvision.datasets.MNIST(root=self.root, train=self.train, download=self.download, transform = transforms.Compose([transforms.ToTensor(),]))
        elif self.dataset_name == "fmnist":
            dataset = torchvision.datasets.FashionMNIST(root=self.root, train=self.train, download=self.download, transform = transforms.Compose([transforms.ToTensor(),]))        
        else:
            raise ValueError("Dataset not found")

        split_number = self.split_number 
         
        #generate spilt dataset
        os.makedirs(f'{self.root}/dat/{self.dataset_name}/{"iid" if self.iid else "noniid"}', exist_ok=True)
        

        if self.dataset_name == "cifar10":
            self.n_cls = 10
            data_len = 50000
            self.channels = 3; self.width = 32; self.height = 32;
        elif self.dataset_name == "cifar100":
            self.n_cls = 100
            data_len = 50000
            self.channels = 3; self.width = 32; self.height = 32;
        elif self.dataset_name == "mnist":
            self.n_cls = 10
            data_len = 60000
            self.channels = 1; self.width = 28; self.height = 28; 
        elif self.dataset_name == "fmnist":
            self.n_cls = 10
            data_len = 60000
            self.channels = 1; self.width = 28; self.height = 28;                                                                                 

        trn_x =dataset.data; trn_y =dataset.targets
                
        self.trn_x = trn_x
        self.trn_y = trn_y
        
        # Init the list of data index
        idx_list = [np.array([], dtype=int) for _ in range(self.n_cls)]

        for i, dat in enumerate(trn_y):
            idx_list[dat] = np.append(idx_list[dat], i)
        if self.sharding ==0:
            clnt_x, clnt_y = self.unbal_n_diri_dist(split_number, data_len, trn_x, trn_y, idx_list)
        else:
            clnt_x, clnt_y = self.sharding_dist(split_number, data_len, trn_x, trn_y, idx_list)

    
        print('Class frequencies:')
        count = 0
        for clnt in range(split_number):
            print("Client %3d: " %clnt + 
                ', '.join(["%.3d" %np.sum(clnt_y[clnt]==cls) for cls in range(self.n_cls)]) + 
                ', Amount:%d' %clnt_y[clnt].shape[0])
            count += clnt_y[clnt].shape[0]
    
    
        print('Total Amount:%d' %count)
        print('--------')
        i=0
        if (len(np.shape(clnt_y[0]))==2):
            self.targets = clnt_y[0].T.tolist()[0]
        else:
            self.targets = clnt_y[0]
        #print(self.targets)
        for clnt_xi, clnt_yi in zip(clnt_x, clnt_y):
            
            fp=open(f'{self.root}/dat/{self.dataset_name}/{"iid" if self.iid else "noniid"}/{i}', 'wb')
            
            pickle.dump((clnt_xi, clnt_yi), fp)
            fp.close()
            i+=1

    def unbal_n_diri_dist(self, split_number, data_len, trn_x, trn_y, idx_list):
        # In our experiments, We don't use unbalanced data.
        
        #unbalaced mount   
        n_data_per_clnt = int(data_len) / self.split_number

        if self.unbalanced_sgm != 0:
            # Get samples from log distribution
            clnt_data_list = (np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalanced_sgm, size=self.split_number))
            clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(trn_y)).astype(int)
            diff = np.sum(clnt_data_list) - len(trn_y)

            if diff != 0:
                for clnt_i in range(split_number):
                    if clnt_data_list[clnt_i] > diff:
                        clnt_data_list[clnt_i] -= diff
                        break

        else:
            # balanced case
            clnt_data_list = (np.ones(split_number) * n_data_per_clnt).astype(int)

        cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]

        # [..., ( the number of Nth client's data X height X width ), ...] 
        if "mnist" in self.dataset_name: # if mnist, fmnist, emnist
            clnt_x = [np.zeros((clnt_data_list[clnt__], self.height, self.width)).astype(np.uint8) for clnt__ in range(split_number)]
            clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(split_number)]                
        else:        
            clnt_x = [np.zeros((clnt_data_list[clnt__], self.height, self.width, self.channels)).astype(np.uint8) for clnt__ in range(split_number)]
            clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(split_number)]
        
        clnt_x, clnt_y = self._dirichlet_dist(split_number, trn_x, trn_y, clnt_data_list, idx_list, cls_amount, clnt_x, clnt_y)
    
        return clnt_x,clnt_y

    def _dirichlet_dist(self, split_number, trn_x, trn_y, clnt_data_list, idx_list, cls_amount, clnt_x, clnt_y):
        # noniid data setting by dirichlet distribution
        
        # Generate prior probabilities and cumulative probabilities for each class
        cls_priors = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls, size=split_number)
        prior_cumsum = np.cumsum(cls_priors, axis=1)


        # Data distribution
        while(np.sum(clnt_data_list) != 0):
            curr_clnt = np.random.randint(split_number)
            # If current node is full, resample a client
            print('Remaining Data: %d' %np.sum(clnt_data_list))
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                if cls_amount[cls_label] <= 0:
                    continue
                cls_amount[cls_label] -= 1
                clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]
                break

        # Convert client data
        clnt_x = np.asarray(clnt_x, dtype=object)
        clnt_y = np.asarray(clnt_y, dtype=object)

        # Calculate the mean for each class
        cls_means = np.zeros((split_number, self.n_cls))
        for clnt in range(split_number):
            for cls in range(self.n_cls):
                cls_means[clnt, cls] = np.mean(clnt_y[clnt] == cls)
        prior_real_diff = np.abs(cls_means - cls_priors)

        # Output results
        print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
        print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))
        return clnt_x,clnt_y
        

    def sharding_dist(self, split_number, data_len, trn_x, trn_y, idx_list):
        # noniid data setting by shard partitioning
        
        total_shard_num = split_number * self.sharding
        each_shard_num = max(total_shard_num / self.n_cls, 2)
        #cls_datalen = int((data_len / self.n_cls) / each_shard_num)
        real_cls_datalen = [int(len(n) / each_shard_num) for n in idx_list]
        #real_cls_datalen = [int((data_len / self.n_cls) / self.sharding) for n in idx_list]

        shard_x = [[] for _ in range(self.n_cls)]
        shard_y = [[] for _ in range(self.n_cls)]

        for label in range(self.n_cls):
            for i in idx_list[label]:
                shard_x[label].append(trn_x[i])
                shard_y[label].append(trn_y[i])

        shard_sel = [each_shard_num for _ in range(self.n_cls)]
        shard_sel_list=[[] for _ in range(split_number)]

        remain_class = list(range(self.n_cls))
        for clnt in range(split_number):
            
            for _ in range(self.sharding):
                while True:
                    label = np.random.choice(remain_class)
                    if shard_sel[label] > 0:
                        shard_sel[label] -= 1
                        shard_sel_list[clnt].append(label)
                        break 
                    remain_class.remove(label)  
        clnt_x, clnt_y =[], []

        print("The list of classes to be patitioned to each node:\n",shard_sel_list)
        for clnt in range(split_number):
            clnt_datalen = 0
            for label in shard_sel_list[clnt]:
                clnt_datalen += real_cls_datalen[label]
            if 'mnist' in self.dataset_name:
                clnt_x += [np.zeros((clnt_datalen, self.height, self.width)).astype(np.uint8)]
                clnt_y += [np.zeros((clnt_datalen)).astype(np.int64)]
            else:
                clnt_x += [np.zeros((clnt_datalen, self.height, self.width, self.channels)).astype(np.uint8)]
                clnt_y += [np.zeros((clnt_datalen)).astype(np.int64)]


        for clnt in range(split_number):
            idx=0
            for _, label in enumerate(shard_sel_list[clnt]):
                cls_datalen= real_cls_datalen[label]
                rest=len(np.array(shard_x[label]))
                if rest<cls_datalen:
                    print(rest)
                    clnt_x[clnt][idx:idx+rest] = np.array(shard_x[label])[:rest]
                    clnt_y[clnt][idx:idx+rest] = np.array(shard_y[label])[:rest]
                    del shard_x[label][:rest]
                    del shard_y[label][:rest]
                    idx+=rest
                else:
                    clnt_x[clnt][idx:idx+cls_datalen] = np.array(shard_x[label])[:cls_datalen]
                    clnt_y[clnt][idx:idx+cls_datalen] = np.array(shard_y[label])[:cls_datalen]
                    del shard_x[label][:cls_datalen]
                    del shard_y[label][:cls_datalen]
                    idx+=cls_datalen

        clnt_x = np.asarray(clnt_x, dtype=object)
        clnt_y = np.asarray(clnt_y, dtype=object)

        return clnt_x, clnt_y


if __name__ =="__main__":   
    root = os.path.abspath('./data')   

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir","-R", help="data directory for load/save the data", type=str, default=root)
    parser.add_argument("--dataset", "-D", help="dataset", type=str, choices=["cifar10", "cifar100", "mnist", "fmnist"], default="mnist")
    parser.add_argument("--alpha","-A", help="alpha of dirichlet distribution", type=float, default=0)
    parser.add_argument("--shard","-S", help="number of classes in a shard", type=int, default=2)
    parser.add_argument("--node","-N", help="number of nodes", type=int, default=80)

    args = parser.parse_args()
    if args.alpha> 10 and args.shard==0:
        iid=True
    else:
        iid=False
    
    custom_dataset(root=args.dir, dataset_name=args.dataset, rule_arg=args.alpha, sharding=args.shard, split_number=args.node, iid=iid)