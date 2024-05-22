import os.path
import pickle
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
class custom_dataset(VisionDataset):
    """
    Get dataset from local dir. Datasets are serialized by pickle
    This Dataset code is only used for vision dataset, so Shakespeare dataset bypass this.
    """

    def __init__(
        self,
        root:str,
        train=True,
        transform=None,
        target_transform=None,
        download=True,
        split_number=10,
        split_id=0,
        dataset_name="cifar10",
    ):
        super(custom_dataset, self).__init__(root=root, transform=transform, target_transform=target_transform)
        self.dataset_name = dataset_name
        self.train = train
        self.root =root
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.split_number = split_number
        self.split_id = split_id
        self.data = []
        self.targets = []
        self.data, self.targets = self.get_data()
        print('Split num : [%d / %d]' %(split_id, split_number))
        
    def __getitem__(self, index):
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_data(self):
        """get data from local dir
        """
        #shakespeare use other code
        if self.dataset_name == "shakespeare":
            return
            
        if self.train == False:
            if os.path.isfile('%s/dat/test' %(self.root)):
                fp=open('%s/dat/test' %(self.root), 'rb')
                dat=pickle.load(fp)
                fp.close()
                tst_x, tst_y=dat
            else:
                raise FileNotFoundError("The data must be in local dir. Please Check %s/dat/test/" %(self.root))
 
            self.data = tst_x; self.targets = tst_y
            return self.data, self.targets
                 
        else:
            if os.path.isfile('%s/dat/train/%d' %(self.root, self.split_id)):
                fp=open('%s/dat/train/%d' %(self.root, self.split_id), 'rb')
                dat=pickle.load(fp)
                fp.close()
                clnt_xi, clnt_yi=dat
                self.data = list(clnt_xi)
                if (len(np.shape(clnt_yi))==2):
                    self.targets = clnt_yi.T.tolist()[0]
                else:
                    self.targets = clnt_yi
                del clnt_xi
                del clnt_yi
                del dat                
            else:
                raise FileNotFoundError("The data must be in local dir. Please Check %s/dat/train/" %(self.root))
            return self.data, self.targets