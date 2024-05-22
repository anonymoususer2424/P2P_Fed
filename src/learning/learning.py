import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import copy
from torch.nn.utils.clip_grad import clip_grad_norm
from .datasets.pickle_dataset import PickleDataset
try: 
    from ._custom_dataloader import *
    from ._merge_param import *
except:
    from _custom_dataloader import *
    from _merge_param import *
from .. import fednode
try:
    from models import *
except:
    from ..models import *

class ABClearning:
    
    def __init__(self, node):
        self.timeval = 0
        self.prev_t=time.time()
        self.alive = True
        self.par = None
        self.dataconfig = node.data_config
        self.device = node.device if torch.cuda.is_available() else 'cpu'
        self.datasets=node.datasets
        self.model=node.model
        self.lr=0.01
        self.local_epoch=5
        self.train=self._train
        self.config_dataset()
        self.config_model()
        self.rounds=0
        self.sleeptime=node.sleeptime 
        self.net = self.net.to(self.device)
        if self.device!='cpu':
            cudnn.benchmark = True
        self.criterion = nn.CrossEntropyLoss()
        self.prev_model=None
        self.mu=0
        self.push = None
    # Training
    def _train(self, epoch):
        net, trainloader, device, optimizer, criterion = self.net, self.trainloader, self.device, self.optimizer, self.criterion
        print('\nEpoch: %d Round: %d' % (epoch, self.rounds))
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        batch_idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            torch.autograd.set_detect_anomaly(True) # type: ignore
            if loss.isnan(): 
                print("loss is nan | batch_idx")
            if self.mu>0:
                self._proximal(net, loss)
                
            loss.backward()
            clip_grad_norm(net.parameters(), 10)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print("train loss : %0.4f  train acc : %0.2f" %(train_loss/(batch_idx+1),100.*correct/total))

    def train_lstm(self, epoch):
        model, train_loader, device, optimizer, criterion = self.net, self.trainloader, self.device, self.optimizer, self.criterion
        total_step = len(train_loader)
        correct_predictions = 0
        total_correct_predictions = 0
        total_predictions = 0
        for i, (data, targets) in enumerate(train_loader):
            # Move tensors to the configured device
            data = data.to(device)
            targets = targets.to(device)
            model.train()
            # Forward pass
            outputs = model(data)

            # Compute loss
            loss = criterion(outputs, targets)

            if self.mu>0:
                self._proximal(model, loss)

            # Backward and optimize (only in training mode)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions = torch.sum(predicted == targets).item()
            total_correct_predictions += correct_predictions
            total_predictions += targets.size(0)

        epoch_accuracy = total_correct_predictions / total_predictions
        print(f'Epoch [{epoch+1}/{self.local_epoch}], Train Accuracy: {epoch_accuracy:.4f}')

    def _proximal(self, model, loss):
        # proximal term for Fedprox
        prox = 0.
        if self.prev_model!=None:
                    # for name, param in net.named_parameters():
            for w0, w in zip(self.prev_model.parameters(), model.parameters()):
                prox += torch.square((w - w0).norm(2))
                        
            loss += self.mu * (0.5 * prox)

    def test_lstm(self):
        net, testloader, device, criterion = self.net, self.testloader, self.device, self.criterion
        test_loss = 0
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            self.test_acc = 100.*correct/total
            print("test  loss : %0.4f  test acc : %0.2f" %(test_loss/(batch_idx+1), self.test_acc))
            
    def run(self, node):
        pass
   
    def config_model(self):
        if self.model=="resnet18":
            self.net = ResNet18(self.num_class) # type: ignore
        elif self.model=="cnn":
            if (self.datasets=="cifar100"):
                print("Dataset and Model might not work!")
            elif (self.datasets=="cifar10"):
                self.net = ComplexCNN() # type: ignore
            else:
                self.net = SimpleCNN(self.num_class) # type: ignore
        elif self.model=="lstm":
            self.net = RNN_Shakespeare() # type: ignore
        else:
            print("Not implemented or Worng model name!")
            assert(NotImplementedError)
            
        if self.model!="lstm":
            self.optimizer = optim.SGD(self.net.parameters(),lr=self.lr,momentum=0.9, weight_decay=5e-4)
        else:
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)

    def config_dataset(self):
        if (self.datasets=="cifar10"):
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            trainset = custom_dataset( # type: ignore
                root='./data', train=True, download=True, transform=transform_train, 
                split_number=self.dataconfig[0], split_id=self.dataconfig[1], dataset_name="cifar10" )
            self.trainloader = data.DataLoader(
                trainset, batch_size=64, shuffle=True)
            self.datalen = len(trainset)
            self.num_class=10
        elif (self.datasets=="cifar100"):
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset = custom_dataset( # type: ignore
                root='./data', train=True, download=True, transform=transform_train, 
                split_number=self.dataconfig[0], split_id=self.dataconfig[1], dataset_name="cifar100" )
            self.trainloader = data.DataLoader(
                trainset, batch_size=64, shuffle=True)
            self.datalen = len(trainset)
            self.num_class=100
        elif (self.datasets=="tinyimagenet200"):
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            trainset = custom_dataset( # type: ignore
                root='./data', train=True, download=True, transform=transform_train, 
                split_number=self.dataconfig[0], split_id=self.dataconfig[1], dataset_name="tinyimagenet200" )
            self.trainloader = data.DataLoader(
                trainset, batch_size=64, shuffle=True)
            self.datalen = len(trainset)
            self.num_class=200
            
        elif (self.datasets=="mnist"):
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            trainset = custom_dataset( # type: ignore
                root='./data', train=True, download=True, transform=transform_train, 
                split_number=self.dataconfig[0], split_id=self.dataconfig[1], dataset_name="mnist" )
            self.trainloader = data.DataLoader(
                trainset, batch_size=64, shuffle=True)
            self.datalen = len(trainset)
            self.num_class = 10
        elif (self.datasets=="fmnist"):
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
            trainset = custom_dataset( # type: ignore
                root='./data', train=True, download=True, transform=transform_train, 
                split_number=self.dataconfig[0], split_id=self.dataconfig[1], dataset_name="fmnist" )
            self.trainloader = data.DataLoader(
                trainset, batch_size=64, shuffle=True)
            self.datalen = len(trainset) 
            self.num_class = 10
        elif (self.datasets=="emnist"):
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1722,), (0.3309,))
            ])
            trainset = custom_dataset( # type: ignore
                root='./data', train=True, download=True, transform=transform_train, 
                split_number=self.dataconfig[0], split_id=self.dataconfig[1], dataset_name="emnist" )
            self.trainloader = data.DataLoader(
                trainset, batch_size=64, shuffle=True)
            self.datalen = len(trainset) 
            self.num_class = 47
        elif (self.datasets=="shakespeare"):
            dataset = PickleDataset(dataset_name='shakespeare')
            train_data = dataset.get_dataset_pickle(dataset_type='train', client_id=self.dataconfig[1])
            test_data = dataset.get_dataset_pickle(dataset_type='test', client_id=self.dataconfig[1])   
            self.trainloader = data.DataLoader(train_data, batch_size=10, shuffle=True)
            self.testloader = data.DataLoader(test_data, batch_size=5, shuffle=True)
            num_class = 80
            self.datalen = len(train_data)
            self.lr=0.4
            self.train=self.train_lstm
        else:
            print("Not implemented or Worng dataset name!")
            assert(NotImplementedError)
