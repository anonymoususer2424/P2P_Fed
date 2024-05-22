from abc import ABC, abstractmethod
import socket, time
import threading
import pickle
import selectors
from src.learning._merge_param import *
import random
from src.learning._custom_dataloader import *
import torch.utils.data as data
from src.models import *
import subprocess

class ABCServer(ABC):

    def __init__(self, addr:tuple, **kwargs):
        # set network config
        # self socket
        self.addr = addr
        print(self.addr)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(addr)
        self.socket.setblocking(False)
        self.socket.listen(5)
        # set listening daemon
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.socket, selectors.EVENT_READ, self.accept_handler)
        self.listen_t = threading.Thread(target=self.run, daemon=True, name="run")
        self.st=0
        # set fed server config
        self.clients=kwargs['clients']
        self.my_classes=kwargs['output']
        self.select=kwargs['select'] #buffer size
        self.slow=kwargs['straggler']
        if kwargs['mode']!=0:
            self.lock=threading.Lock()
            self.lock1=threading.Lock()
            self.lock3=threading.Lock()
            self.par_list = []
        self.save = 0
        self.net = getmodel[kwargs["model"]](self.my_classes)
        if ( kwargs["model"] == "RNN_Shakespeare" ):
            self.save = 1
        else:
            print("bye bye")
        # ready for manage client
        self.c_list = []
        self.weights = dict()
        self.rounds=0
        self.parQueue = []
        self.fedopt = FedAvg_modif()
        self.savetimes=0
        self.criterion = nn.CrossEntropyLoss()
        self.test_acc = []
         
        print('start') 
        self.alive = True
        self.listen_t.start()
        # default job(for now inf loop)
        self.mainjob()
     
    def mainjob(self):
        # check=False
        while self.alive: 
            # for evaluate traffic cost.
            # for this, check dockerfile and add 'apt install nmon'.
            
            # if (self.rounds == 1 and check==False):
            #     subprocess.Popen(args=["nmon -f -s 1"] ,stdout=subprocess.PIPE, shell=True)
            #     check=True
            # elif (self.rounds == 6 and check==True):
            #     subprocess.Popen(args=["pkill nmon"] ,stdout=subprocess.PIPE, shell=True)
            #     check=False
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
    
    def _handle(self, data, conn: socket.socket):
        """
        handle data from other nodes
        """
        if data[0] == 'join':
            self._join(data)
        
        elif data[0] == 'exit':
            self._exit(data)
        elif data[0] == 'test_chat':
            self._test_chat(data, conn)
            
        elif data[0] == 'push_param':
            self._handle_push_param(data, conn)
        else:
            a=0
            
        if conn:
            conn.close
    
    @abstractmethod
    def _handle_push_param(self, data, conn: socket.socket):
        # need to be define
        pass

    @abstractmethod
    def update_client(self, param=None):
        # need to be define
        pass
    
    def _join(self, data):
        
        new_client = data[1]
        self.c_list.append(new_client)
        print('new client {} joined!\nnow {} clients in list!'.format(new_client, len(self.c_list)))
        
        if len(self.c_list)==self.clients:
            self.st=time.time()
            self.update_client()
    
    def _exit(self, data):
        client = data[1]
        self.c_list.remove(client)
        print('client {} exit!\nnow {} clients in list!'.format(client, len(self.c_list)))
    
    def _test_chat(self, data, conn:socket.socket):
        print(" > ", conn.getpeername(), " says ", data[1])
    
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
        data = conn.recv(2048)
        data = pickle.loads(data)
        threading.Thread(target=self._handle, args=((data,conn)), daemon=True).start()
        sel.unregister(conn)
    
    def save_model(self):
        if not os.path.exists("/workspace/mdls"):
            os.makedirs("/workspace/mdls")
        self.savetimes+=1
        path="/workspace/mdls/{}_0_{}.pt".format(self.clients,self.savetimes)
        torch.save({
            'round' : self.rounds,
            'model_state_dict':self.net.state_dict(),
            'time' : time.time()-self.st
        }, path)
               
def find_free_ports(start_port, end_port, self_addr):
    port=find_free_ports.PORT+self_addr[1]
    try:
        # Create a socket and attempt to bind to the port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self_addr[0], port))
            find_free_ports.PORT+=1
            if(find_free_ports.PORT>=50):
                find_free_ports.PORT=1
            return port
    except:
        find_free_ports.PORT+=1
        pass  # Port is already in use
    raise ConnectionError("cannot find free port")
find_free_ports.PORT=1
