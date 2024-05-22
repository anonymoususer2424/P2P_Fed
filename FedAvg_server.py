import socket, time, pickle, random
from ABCServer import ABCServer
import torch
import subprocess
import argparse
import threading
import copy

class FedAvg_server(ABCServer):
    def __init__(self, addr:tuple, **kwargs):
        ratio=kwargs['ratio']
        kwargs['select']=max(round(ratio*kwargs['clients']*0.01),1)
        super().__init__(addr, **kwargs)
                    
    def _handle_push_param(self, data, conn: socket.socket):
        size = data[1]
        datalen = data[3]
        print("<", data[2])
        conn.send(b'ok')
        if self.save:
            self.test_acc.append(data[5])
        else:
            pass
        #receive param                   
        data=[]
        while size>0:
            s = conn.recv(size)
            if not s: break
            data.append(s)
            size -= len(s)
        data = b''.join(data)
        
        conn.close()

        data = pickle.loads(data)
        self.weights[conn.getpeername] = data
        device="cpu"
        self.parQueue.append((0, datalen, data))
        if len(self.weights) >= self.curr_select:
            
            if self.save:
                test_avg = sum(float(x) for x in self.test_acc) / len(self.test_acc)
                print("average test acc : %0.2f" %(test_avg))
            else:
                pass
            
            par = list(self.weights.values())[0]
            self.weights = dict()
            print("received %d weights, now round %d" %(len(par),self.rounds))
            print("accumulated time : {:.3f}".format(time.time()-self.st))
            #print("average test acc : %0.2f" %(test_avg))
            self.rounds+=1
            self.test_acc = []
            agg_t = time.time()
            new_param = self.fedopt.do(par=par, P=self.parQueue,dev='cpu')
            print("aggregate with {:.4f}".format(time.time()-agg_t))
            self.parQueue = []
            self.net.load_state_dict(new_param)
            self.net = self.net.to(torch.device(device))
            st=time.time()
            self.update_client(new_param)
            print("send time : ", time.time()-st)
            if self.rounds%5==1:
                tt=time.time()
                self.save_model()
                print("save time : ", time.time()-tt)
            
    def update_client(self, param=None):
        while(True):
            clients = random.sample(self.c_list,self.select)
            print(self.slow)
            
            if param is not None:
                par = pickle.dumps(param)
            
            self.curr_select = self.select
            for c in clients:
                # if time.time()-self.st>100 and 1000>time.time()-self.st and c in self.c_list[int(0.9*self.clients):]:
                #     self.curr_select-=1
                #     continue
                slow=False
                if self.c_list.index(c)<self.slow:
                    slow=True
                print(c, slow)
                    
                sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                port_ =find_free_ports(self.addr[1]+1,self.addr[1]+49,  self.addr)
                sock.bind((self.addr[0], port_))
                
                sock.connect(c)
                if param==None:
                    sock.send(pickle.dumps(('_start_learning',{'slow':slow})))
                else:
                    sock.send(pickle.dumps(('_weight', len(par), self.rounds, {'slow':slow})))
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    if sock.recv(1024).decode('utf-8')=='ok':
                        sock.send(par)

                sock.close()
            if self.curr_select==0:
                print("skip this round")
            else:
                break

class FedBuff_server(ABCServer):
    def __init__(self, addr:tuple, **kwargs):
        kwargs['select']=3
        super().__init__(addr, **kwargs)
        self.pre = [None,None]
        self.save = super().save
    def _handle_push_param(self, data, conn: socket.socket):
        #print("start")
        size = data[1]
        datalen = data[3]
        self.lock1.acquire()
        self.par_list.append(data[2]) #save client
        if self.save:
            self.test_acc.append(data[5])
        rounds = data[4]
        print("<", data[2])
        conn.send(b'ok')
        self.lock1.release()
        #receive param                   
        data=[]
        while size>0:
            s = conn.recv(size)
            if not s: break
            data.append(s)
            size -= len(s)
        data = b''.join(data)
        
        conn.close()

        data = pickle.loads(data)
        #self.weights[conn.getpeername] = data
        device="cpu"
        print("received %d weights" %(len(self.par_list)))
        
        self.lock.acquire()
        self.parQueue.append((0, datalen, data, rounds))
        #self.lock.release()
        if len(self.parQueue) >= self.select:
            par = self.parQueue[0][2]          
            agg_t = time.time()
            #self.lock.acquire()
            if self.rounds == 0:
                new_param, self.pre = self.fedopt.do(par=par, P=self.parQueue[:self.select], dev='cpu', pre=[None,None])
            else:
                new_param, self.pre = self.fedopt.do(par=par, P=self.parQueue[:self.select]+[(0, self.pre[0], self.pre[1], self.rounds-1)] ,dev='cpu', pre=self.pre)
            
            print("aggregate with {:.4f}".format(time.time()-agg_t))
            del self.parQueue[:self.select]
            self.pre_data = new_param
            self.lock.release()

            self.net.load_state_dict(new_param)
            self.net = self.net.to(torch.device(device))
            self.update_client(new_param)
            
            if self.save == 0:
                if self.rounds%10==1:
                    if self.save:
                        return
                    tt=time.time()
                    self.save_model()
                    print("save time : ", time.time()-tt, "at round", self.rounds)
        else:
            self.lock.release()
                
    def update_client(self, param=None):
        clients = self.c_list

        self.lock1.acquire()

        if param is not None:
            par = pickle.dumps(param)
            clients = copy.deepcopy(self.par_list[:self.select])
            num=self.select
            del self.par_list[:num] 
            
            if self.test_acc:
                test_avg = sum(float(x) for x in self.test_acc[:self.select]) / num
                print("average test acc : %0.2f" %(test_avg))
                self.save = 1
            else:
                pass
            del self.test_acc[:num]
            self.rounds+=1 
            print("received %d weights, now round %d" %(len(par),self.rounds))
            print("accumulated time : {:.3f}".format(time.time()-self.st)) 
        self.lock1.release()
        
        self.lock3.acquire()        
        for c in clients:
            print(c)
            slow=False
            if self.c_list.index(c)<self.slow:
                slow=True

            sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port_ =find_free_ports(self.addr[1]+1,self.addr[1]+49,  self.addr)
            sock.bind((self.addr[0], port_))
            
            sock.connect(c)
            if param==None:
                sock.send(pickle.dumps(('_start_learning',{'slow':slow})))
            else:
                sock.send(pickle.dumps(('_weight', len(par), self.rounds,{'slow':slow})))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if sock.recv(1024).decode('utf-8')=='ok':
                    sock.send(par)
            sock.close()
            
          
        self.lock3.release()

class FedAsync_server(ABCServer):
    def __init__(self, addr:tuple, **kwargs):
        kwargs['select']=1
        #self.alpha=0.9 # for example and default, α = 0.9, ploy a = 0.5
        self.alpha=kwargs['alpha']
        self.shed_time=kwargs['scheduler']
        self.mu=kwargs['proximal']
        self.lock2=threading.Lock()
        super().__init__(addr, **kwargs)
    
    def mainjob(self):
        self.update_g = threading.Thread(target=self.update_global, daemon=True)
        self.update_c = threading.Thread(target=self.scheduler, daemon=True)
        
        self.update_g.start()
        self.update_c.start()
        
        return super().mainjob()
    def _handle_push_param(self, data, conn: socket.socket):
        size = data[1]
        datalen = data[3]
        self.lock1.acquire()
        self.par_list.append(data[2]) #save client
        print("received %d weights" %(len(self.par_list)))
        print("<", data[2])
        self.lock1.release()
        #print("<", data[2])
        rounds = data[4]
        conn.send(b'ok')
        if self.save:
            print("test acc : ",data[5])
            
        print("accumulated time : {:.3f}".format(time.time()-self.st))    
        #receive param                   
        data=[]
        while size>0:
            s = conn.recv(size)
            if not s: break
            data.append(s)
            size -= len(s)
        data = b''.join(data)
        conn.close()
        data = pickle.loads(data)
        #self.weights[conn.getpeername] = data
        device="cpu"
        #print("received %d weights" %(len(self.par_list)))
        
        self.lock.acquire()
        self.parQueue.append((0, datalen, data,rounds))
        self.lock.release()
    
    def update_global(self):
        bat = 'batch'
        mean= 'mean'
        var = 'var'
        while True:
            #time.sleep(0.1)
            if self.st!=0:
                self.lock.acquire()
                
                if len(self.parQueue)>0:
                    agg_weights = copy.deepcopy(self.parQueue[0][2])
                    global_par = self.net.state_dict()
                    tau= self.parQueue[0][3]
                    #self.lock.release()
                    t = self.rounds
                    at=self.alpha * 1/((t-tau+1)**0.5)
                    del self.parQueue[0]
                    self.lock.release()
                    print('tau, t, at',tau, t, at)
                    for k in agg_weights.keys():
                        # if (mean in k) or (var in k) or (bat in k):
                        #     agg_weights[k] = 0.5*par[k]+0.5*global_par[k]
                        # else:
                        agg_weights[k] = at*agg_weights[k]+(1-at)*global_par[k]
                    self.lock2.acquire()
                    self.net.load_state_dict(agg_weights)
                    del agg_weights
                    self.net = self.net.to(torch.device('cpu'))
                    self.rounds+=1
                    self.lock2.release()    
                    if self.rounds%20==1:
                        tt=time.time()
                        self.save_model()
                        print("save time : ", time.time()-tt, "at round", self.rounds)
                else:
                    self.lock.release()

        
    def scheduler(self):
        while True:
            time.sleep(self.shed_time)
            if self.st!=0:
                self.lock2.acquire()
                par=self.net.state_dict()
                self.lock2.release()
                self.update_client(par)
    def update_client(self, param=None):
        clients = self.c_list

        self.lock1.acquire()

        if param is not None:
            par = pickle.dumps(param)
            clients = copy.deepcopy(self.par_list[:])
            del self.par_list[:]
        self.lock1.release()
        for c in clients:
            print(c)
            slow=False
            if self.c_list.index(c)<self.slow:
                slow=True

            sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port_ =find_free_ports(self.addr[1]+1,self.addr[1]+49,  self.addr)
            sock.bind((self.addr[0], port_))
            
            sock.connect(c)
            if param==None:
                sock.send(pickle.dumps(('_start_learning',{'slow':slow,'mu':self.mu})))
            else:
                sock.send(pickle.dumps(('_weight', len(par), self.rounds,{'slow':slow,'mu':self.mu})))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if sock.recv(1024).decode('utf-8')=='ok':
                    sock.send(par)

            sock.close()


class FedProx_server(FedAvg_server):
    def update_client(self, param=None):

        clients = random.sample(self.c_list,self.select)
        
        
        if param is not None:
            par = pickle.dumps(param)
        self.curr_select = self.select
        for c in clients:
            print(c)
            slow=False
            if self.c_list.index(c)<self.slow:
                slow=True

            sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port_ =find_free_ports(self.addr[1]+1,self.addr[1]+49,  self.addr)
            sock.bind((self.addr[0], port_))
            
            sock.connect(c)
            if param==None:
                sock.send(pickle.dumps(('_start_learning',{'mu':1, 'slow':slow})))
            else:
                sock.send(pickle.dumps(('_weight', len(par), self.rounds,{'mu':1, 'slow':slow})))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if sock.recv(1024).decode('utf-8')=='ok':
                    st=time.time()
                    sock.send(par)
                    print("----",time.time()-st)

            sock.close()

def find_free_ports(start_port, end_port, self_addr):
    port=find_free_ports.PORT+self_addr[1]
    try:
        #print(port)
        # Create a socket and attempt to bind to the port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self_addr[0], port))
            #print(s.getsockname())
            #print(port)
            find_free_ports.PORT+=1
            if(find_free_ports.PORT>=50):
                find_free_ports.PORT=1
            return port
    except:
        find_free_ports.PORT+=1
        pass  # Port is already in use
    print("!!!!!!!!!!!!!!!!port is none!!!!!!!!!!!!!!!")
    exit()
    return None
find_free_ports.PORT=1

def get_global_ip():
    """
    get global ip address
    """
    return subprocess.check_output("wget http://ipecho.net/plain -O - -q ; echo", shell=True).decode().strip()

def get_self_ip():
    return socket.gethostbyname(socket.gethostname())

if __name__ == '__main__':
    this_ip = get_self_ip()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port","-p", help="server's port number", type=int, default=16000)
    parser.add_argument("--ip","-a", help="server's ip address", type=str, default=this_ip)
    parser.add_argument("--clients","-c", help="number of client", type=int, default=10)
    parser.add_argument("--ratio","-r", help="selection ratio(ignored when Async)", type=int, default=10)
    parser.add_argument("--output","-o", help="data classes", type=int, default=10)
    parser.add_argument("--mode","-A", help="async setting when 0: sync(default), 1: buff, 2: async", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--model", "-M", help="model to train", type=str, choices=["ResNet18", "ComplexCNN", "SimpleCNN", "RNN_Shakespeare"], default="ResNet18")
    parser.add_argument("--straggler", "-s", help="number of straggler(slow node)", type=int, default=0)
    parser.add_argument("--proximal", "-m", help="temporary", type=float, default=0)
    parser.add_argument("--alpha", help="temporary", type=float, default=1)
    parser.add_argument("--scheduler", help="temporary", type=float, default=0)
    
    args = parser.parse_args()
    
    this_addr = (args.ip, args.port)
    
    if args.mode == 0:
        print("FedAvg\nconfig :",dict(args._get_kwargs()))
        server = FedAvg_server(this_addr, **dict(args._get_kwargs()))
        
    elif args.mode == 1:
        print("FedBuff\nconfig :",dict(args._get_kwargs()))
        server = FedBuff_server(this_addr, **dict(args._get_kwargs()))
        
    elif args.mode == 2:
        print("FedAsync\nconfig :",dict(args._get_kwargs()))
        server = FedAsync_server(this_addr, **dict(args._get_kwargs()))
    
    elif args.mode == 3:
        print("FedProx\nconfig :",dict(args._get_kwargs()))
        server = FedProx_server(this_addr, **dict(args._get_kwargs()))
