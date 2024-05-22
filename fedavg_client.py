import socket, time
import threading
import pickle
import selectors
import subprocess
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.nn.utils.clip_grad import clip_grad_norm
from src.models import *
from src.learning._custom_dataloader import *
from src.utils.utils import get_global_ip, get_self_ip, find_free_ports
from datasets.pickle_dataset import PickleDataset
from src.learning.learning import *

class fedclient:
    
    def __init__(self, addr:tuple, hostaddr:tuple, data_config, device, iid, model, datasets, **kwargs):
        
        self.alive = True
        
        self.mode = False
        # daemon listening
        self.addr = addr
        self.host_addr = hostaddr
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.realaddr = (get_self_ip(), addr[1])
        self.socket.bind(self.realaddr)
        self.socket.setblocking(False)
        self.socket.listen(5)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.socket, selectors.EVENT_READ, self.accept_handler)
        self.listen_t = threading.Thread(target=self.run, daemon=True, name="run")
        self.listen_t.start()
        self.data_config = data_config
        self.device = device
        self.iid = iid
        self.rounds = 0
        # import learning
        self.model = model
        self.datasets = datasets
        self.sleeptime=6
        self.learning= Learning(self)

        # notify self
        while 1:
            try:
                self.join()
                break
            except(ConnectionError):
                print("Connection Error start again..")
                time.sleep(1)
                pass
        
        #interactive mode for test maybe default setting or change 
        #this may become a default or may be replaced by a sending a heartbeat.
        if self.mode:
            dat = input("input the test str\n")
            while dat!='exit':
                self.send(dat)
                dat = input("input the test str\n")
            
            self.exit()
        else:
            while(self.alive):
                time.sleep(5)   
            
    def run(self):
        """
        thread for listening
        """
        while self.alive:
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.socket, selectors.EVENT_READ, self.accept_handler)
            while self.alive:
                for (key,mask) in self.selector.select():
                    key: selectors.SelectorKey
                    srv_sock, callback = key.fileobj, key.data
                    callback(srv_sock, self.selector)

    def join(self):
        sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_ =find_free_ports(self.addr[1]+1,self.addr[1]+49,  self.realaddr)
        sock.bind((self.realaddr[0], port_))
        sock.connect(self.host_addr)
        sock.send(pickle.dumps(('join', self.addr)))
        sock.close()
    
    def exit(self):
        sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_ =find_free_ports(self.addr[1]+1,self.addr[1]+49,  self.realaddr)
        sock.bind((self.realaddr[0], port_))
        sock.connect(self.host_addr)
        sock.send(pickle.dumps(('exit', self.addr)))
        sock.close()
            
    def send(self, data):
        sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_ =find_free_ports(self.addr[1]+1,self.addr[1]+49,  self.realaddr)
        sock.bind((self.realaddr[0], port_))
        sock.connect(self.host_addr)
        sock.send(pickle.dumps(('test_chat', data)))
        sock.close()
    
    def _handle(self, data, conn: socket.socket):
        """
        handle data from other nodes
        """
        #print(data)
        #data = pickle.loads(data)
        #self.logger.debug("[recv data : {}]".format(data))

        if data[0] == '_start_learning':
            print("run")
            if len(data)>1: #for additional config (ex, fedprox)
                self.learning.mu=data[1].get('mu', 0)
                self.learning.slow=data[1]['slow']
            self.learning.run(self, None)
        
        elif data[0] == '_weight':
            if len(data)>3: #for additional config (ex, fedprox)
                self.learning.mu=data[3].get('mu', 0)
                self.learning.slow=data[3]['slow']
            self._update_weight(data, conn)
        
        else:
            a=0         
        if conn:
            conn.close    
    
    def _update_weight(self, data, conn):
        size = data[1]
        self.rounds = data[2]
        print("recv", data[1:])
        conn.send(b'ok')

        data=[]
        while size>0:
            s = conn.recv(size)
            #print("----- recv_t {}".format(len(s)))
            if not s: break
            data.append(s)
            size -= len(s)
            #print("               {:.4f}".format(time.time()-temp))
        data = b''.join(data)
        
        conn.close()
        
        data = pickle.loads(data)        
        # for param, _new in zip(net.parameters(), new_param):
        #     param.data = _new
        self.learning.run(self, data)

    ####################### for handling connection ######################
    def accept_handler(self, sock: socket.socket, sel: selectors.BaseSelector):
        """
        accept connection from other nodes
        """
        conn: socket.socket
        conn, addr = sock.accept()
        sel.register(conn, selectors.EVENT_READ, self.read_handler)

    def read_handler(self, conn: socket.socket, sel: selectors.BaseSelector):
        """
        read data from other nodes
        """
        message = "---- wait for recv[any other] from {}".format(conn.getpeername())
        #self.logger.debug(message)  
        data = conn.recv(1024)
        time.sleep(0.5)
        #self._handle(data, conn)
        data = pickle.loads(data)
        threading.Thread(target=self._handle, args=((data,conn)), daemon=True).start()
        sel.unregister(conn)
                
class Learning(ABClearning):
    
    def __init__(self, node):
        self.st = time.time()
        super().__init__(node)
        
        self.push = self.push_param
        self.slow = False
        self.host_addr = node.host_addr
        self.addr = node.addr
        self.realaddr=node.realaddr
         
    def run(self, node, new_param=None):
        self.rounds = node.rounds
        self.st = time.time()
        net = self.net
        if new_param!=None:
            self.net.load_state_dict(new_param)
            self.net = self.net.to(torch.device(self.device))
            self.prev_model=self.net
        if self.slow==True and self.mu>0:
            # fedprox
            e= random.randint(1, self.local_epoch)
        else:
            e= self.local_epoch
            
        for epoch in range(e):
            self.train(epoch)
            end_t = time.time()
            self.timeval=end_t-self.prev_t
            print("epoch time : {:.4f} ({:.4f})".format(time.time()-self.st, self.timeval))
            if self.slow==True:
                # for hetero devide setting
                print("sleep")
                time.sleep(self.sleeptime)

            self.prev_t = time.time()

        if (self.datasets=="shakespeare"):
            tt = time.time()
            self.test_lstm()
            print("val time : {:.4f}".format(time.time()-tt))
            print("Time : {:.4f}".format(time.time()-self.st))

        self.par = self.net.to(torch.device("cpu")).state_dict()
        self.net.to(torch.device(self.device))

        self.push()
            
    def push_param(self):
            
            par = pickle.dumps(self.par)
            
            sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port_ =find_free_ports(self.addr[1]+1,self.addr[1]+49,  self.realaddr)
            sock.bind((self.realaddr[0], port_))
            
            sock.connect(self.host_addr)
            if (self.datasets=="shakespeare"):
                sock.send(pickle.dumps(('push_param', len(par), self.addr, self.datalen,  self.rounds, "%.2f" %self.test_acc)))
            else:
                sock.send(pickle.dumps(('push_param', len(par), self.addr, self.datalen, self.rounds)))

            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if sock.recv(1024).decode('utf-8')=='ok':

                sock.send(par)
            sock.close()

if __name__ == '__main__':
    this_ip = get_self_ip()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port","-p", help="this peer's port number", type=int, default=random.randint(16001, 40000))
    parser.add_argument("--addr","-a", help="this peer's ip address", type=str, default=this_ip)
    parser.add_argument("--host_port","-P", help="help peer's port number", type=int, default=16000)
    parser.add_argument("--host_addr","-A", help="help peer's ip address", type=str, default='220.67.133.165')
    #parser.add_argument('--test', '-t', help="option to test", action="store_true", default=False)
    
    parser.add_argument("--data","-t", help="use (N)'th block in data", type=int, default=0)
    
    parser.add_argument("--gpu","-g", help="gpu_num", type=int, default=0)
    parser.add_argument("--clients","-c", help="number of client", type=int, default=10)
    parser.add_argument("--iid","-i", help="non-iid data", action="store_false", default=True)   
    parser.add_argument("--model","-m",  help="model's name(resnet18, cnn)", type=str, default="resnet18")
    parser.add_argument("--dataset","-d", help="dataset's name(cifar10, cifar100, tinyimagenet200)", type=str, default="cifar10")
    
    args = parser.parse_args()
    
    if args.gpu == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(args.gpu)
    
    

    if args.data in range(args.clients):
        case = [args.clients,args.data]
    else:
        raise ValueError("case_n must be less than 6")
    
    this_addr = (args.addr, args.port)
    host_addr = (args.host_addr, args.host_port)
    realaddr = (get_self_ip(), this_addr[1])
    print(this_addr, host_addr, realaddr)
    client = fedclient(this_addr, host_addr, case, device, args.iid, args.model, args.dataset)
    
