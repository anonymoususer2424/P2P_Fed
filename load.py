import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import os
import sys
from os.path import abspath, dirname
current_file_path = abspath(__file__)
current_directory = dirname(current_file_path)
project_root = abspath(dirname(dirname(current_directory)))
sys.path.append(project_root)
#from src import fednode
from src.models import *
#from src.learning._custom_dataloader import *
import argparse
import concurrent.futures
parser = argparse.ArgumentParser()
parser.add_argument("-g", help="number of GPUs", type=int, default=2)
parser.add_argument("-n", help="number of nodes", type=int, default=16)
parser.add_argument("-u", help="n", type=int, default=16)
parser.add_argument("-m", help="0: fedchord, 1: fedavg", type=int, default=1)
parser.add_argument("-d", help="dataset to use", type=str, default="mnist")
args = parser.parse_args()

if len(sys.argv) == 1:
    print("Please provide arguments. Use -h for help.")
    sys.exit(1)
elif args.g is None or args.n is None:
    print("Please provide both -g (number of GPUs) and -n (number of nodes) arguments.")
    sys.exit(1)
    
num_gpus = args.g
num_nodes = args.n
gid = args.u
if args.m==0:
    mode = "DFL_log"
else:
    mode = "CFL_log"
#print(num_gpus)
curr_day = time.strftime("%d")
curr_time = time.strftime("%H:%M") 
def test(node_id, net, testloader, criterion):
    device = f"cuda:{gid}"
    net=net.to(device)

    round_num = 1
    while True:
        file_path = f"./model/mdls/{num_nodes}_{node_id}_{round_num}.pt"
        if not os.path.exists(file_path):
            break

        checkpoint = torch.load(file_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        rounds = checkpoint['round']
        check_time = checkpoint['time']


        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        batch_idx = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            accuracy = 100. * correct / total
            dir_path = f"./log/{mode}/{curr_day}/{curr_time}"
            log_file_path = os.path.join(dir_path, f"ct{node_id}.txt")
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

            with open(log_file_path, 'a') as file:
                #file.write(f"Node ID: {node_id}\n")
                file.write(f"round: {rounds}\n")
                file.write(f"time: {round(check_time, 2)}  ")
                file.write(f"loss: {test_loss/(batch_idx+1):.3f}  ")
                file.write(f"test acc: {accuracy:.2f}\n")
        round_num += 1

def get(gid):    
    cudnn.benchmark = True
    
    if args.d=="mnist":
        net = SimpleCNN(10)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)), #mnist
        ])
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform_test)
    elif args.d=="fmnist":
        net = SimpleCNN(10)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)), #fmnist 
        ])
        testset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform_test)
    elif args.d=="emnist":
        net = SimpleCNN(47)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1722,), (0.3309,)), #emnist
        ])
        testset = torchvision.datasets.EMNIST(
            root='.d/data',split='balanced', train=False, download=True, transform=transform_test)
    elif args.d=="cifar10":
        net = ResNet18(10)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #CIFAR
        ])
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    elif args.d=="cifar100":
        net = ResNet18(100)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #CIFAR
        ])
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
    criterion = nn.CrossEntropyLoss()

    testloader = data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    return net, testloader, criterion

def main():
    st_1 = time.time()
    net, testloader, loss= get(gid)
    for i in range(num_nodes):
        if i%num_gpus==gid:
            test(i,  net, testloader, loss)
    print("time : {:.4f}".format(time.time()-st_1))        
if __name__ == "__main__":
    main()
