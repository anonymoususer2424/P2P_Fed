try:
    from . import node
except:
    import node
from .learning.learning import ABClearning
from .utils.utils import find_free_ports
from copy import deepcopy
import threading
import traceback
import socket, time,pickle
from .learning import *
    
__all__ = ['fednode']

class fednode(node.P2PNode):
    def __init__(self, config, logger, addr, host_addr=None, device=None, data_config:list=[1,0]):
        ", model, datasets, container=False, iid=True"
        print("         fednode - OK")
        self.st = time.time()
        self.lock=threading.Lock()
        self.device = device
        self.data_config = data_config
        self.iid = config["iid"]
        self.model = config["model"]
        self.datasets = config["dataset"]
        self.slow = True if config["slow"] > data_config[1] else False
        print(self.slow)
        self.dfedavgm = config["dfedavgm"] #else fedchord
        self.sleeptime = 6
        
        self.learning = self._learning(self)
        self.stable_time = 1
        self.rounds = 0
        super().__init__(logger=logger, 
                         addr=addr, 
                         host_addr=host_addr, 
                         container=config["container"], 
                         data_con=data_config)
              
    def _handle(self, data, conn):
        """
        handle data from other nodes
        """
        if data[0] == 'push_param':
            "the case of push param"
            self._handle_push_param(data,conn)
            #pass 
        super()._handle(data, conn)

    def mainjob(self):
        time.sleep(100)
        self.stable_time=3

        print("start learning")
        self.st=time.time()
        for self.rounds in range(2000):
            self.learning.run(self)
        
    def _handle_push_param(self, para, conn: socket.socket):
        """
        push param to other node
        """
        size = para[1]
        datalen = para[2]
        _id = para[3]

        conn.send(b'ok')
        recv_t = time.time()            
        data=[]
        while size>0:
            temp =time.time()
            s = conn.recv(size)
            if not s: break
            data.append(s)
            size -= len(s)
        data = b''.join(data)
        
        conn.close()
        print('recv parameters time:{:.4f}'.format(time.time()-recv_t))
        unzip_t = time.time()

        data = pickle.loads(data)

        print('deserialize time:{:.4f}'.format(time.time()-unzip_t))
        found=False
        with self.lock:
            if self.dfedavgm!=True:
                for i,par in enumerate(self.learning.parQueue):
                    if par[0]==_id:
                        self.learning.parQueue[i] = (_id, datalen, data)
                        found=True
            if found==False:
                self.learning.parQueue.append((_id, datalen, data))
    class _learning(ABClearning):
        def __init__(self, node):
            self.st = node.st
            self.lock = node.lock
            super().__init__(node)
            self.slow = node.slow
            self.fedopt = FedAvg_modif()
            self.dfedavgm = node.dfedavgm
            
            self.savetimes=0
            self.push = self.default_push
        
        def run(self, node):
            self.st=node.st
            self.addr = node.addr
            self.realaddr = node.realaddr
            self.succ = node.successor_addr
            self.pred = node.predecessor_addr
            self.id = node.id
            self.rounds = node.rounds

            print("round",self.rounds)
            if self.rounds!=0:
                """
                wait until self.par == rounds
                """
                self.parQueue[0]=(self.id, self.datalen, self.par) # type: ignore

                while True:
                    if self.dfedavgm:
                        if (len(self.parQueue)>=3):
                            with self.lock:
                                P=copy.deepcopy(self.parQueue[:3])
                                if len(self.parQueue)>3:
                                    self.parQueue=[self.parQueue[0]]+self.parQueue[3:]
                                else:
                                    self.parQueue=[self.parQueue[0]]
                            break
                        time.sleep(1)
                    else: # P2P-Fed
                        if (len(self.parQueue)>1):
                            with self.lock:
                                P=copy.deepcopy(self.parQueue)
                                self.parQueue=[self.parQueue[0]]
                            break
                        time.sleep(1)
                up_t = time.time()
                agg_t = time.time()
                    
                with torch.no_grad():
                    new_param = self.fedopt.do(par=self.parQueue[0][2], P=P,dev=self.device)
                print("aggregate with {:.4f}".format(time.time()-agg_t))
                self.net.load_state_dict(new_param)
                self.net = self.net.to(torch.device(self.device))
                self.prev_model=self.net
                del P
                if (self.datasets!="shakespeare"):
                    if self.rounds%5==1:
                        tt=time.time()
                        self._save_model()
                        print("save time : ", time.time()-tt)
            else:
                self.parQueue = [(self.id, self.datalen,None),]
                
            print("avg time : {:.4f}".format(time.time()-self.prev_t))
            self.prev_t = time.time()
            for epoch in range(self.local_epoch):
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

            self.finger_table = copy.deepcopy(node.finger_table)
            send_t=time.time()
            self.push()
            print("send time : {:.4f}".format(time.time()-send_t))
            self.prev_t = time.time()
    
            
        def default_push(self):
            seri_t = time.time()
            
            par = pickle.dumps(self.par)
            self._push_param(seri_t, par)

        def _push_param(self, seri_t, par):
            print('serialize the model',end=' ')
            print('time:{:.4f}'.format(time.time()-seri_t))
            send_t = time.time()
            print('R=',self.rounds)
            
            # sampling subset S of peers
            # case of random select
            if self.dfedavgm:
                S=[self.pred, self.succ]
            else: #P2P-Fed
                S=[]
                
                randIds = np.random.choice(range(0,self.dataconfig[0]), 8)*int((2**32)/self.dataconfig[0])
                for _id in randIds:
                    peer=startquery(_id,self.succ,self.realaddr)
                    if peer ==0:
                        print("this time, failed to sample random peer..")
                        break
                    elif peer is not None:
                        S.append(peer)
            S= list(set(S))    
            
            # case of fingertable
            # ft=[]
            # temp_ft = list(set(deepcopy(self.finger_table)))
            # for finger in temp_ft:
            #     ft+=[finger[0]]
            #S= list(set(ft))
            
            threads=[]
            for finger in S:
                t = threading.Thread(target=self._sending_param, args=(par, finger))
                t.start()
                threads.append(t)
            
            for t in threads:
                t.join() 
                
            del par

        def _sending_param(self, par, finger):
            try:
                if(finger==self.addr):
                    print("---- this is me")
                else:
                    print("---- prepare to conn with {}".format(finger))
                    send_t = time.time()
                    sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    port_ = find_free_ports(self.realaddr[1]+1,self.realaddr[1]+49,  self.realaddr)
                    sock.bind((self.realaddr[0], port_))

                        
                    sock.connect(finger)
                    print("---- send meta {}".format(len(par)))

                    sock.send(pickle.dumps(('push_param', len(par), self.datalen, self.id)))

                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    if sock.recv(1024).decode('utf-8')=='ok':
                        print("---- recv ok, send par")
                        send_t = time.time()
                        sock.send(par)

                    else:
                        print("---- rejected..!!!")
                    sock.close()

            except ConnectionRefusedError:
                traceback.print_exc()
                print(finger)
            except Exception:
                traceback.print_exc()
                print(finger)
                
        def _save_model(self):
            if not os.path.exists("mdls"):
                os.makedirs("mdls")
            self.savetimes+=1
            path="mdls/{}_{}_{}.pt".format(self.dataconfig[0], self.dataconfig[1], self.savetimes)
            torch.save({
                'round' : self.rounds,
                'model_state_dict':self.net.state_dict(),
                'time' : time.time()-self.st
            }, path)

def startquery(id_,host_addr,self_addr):

    st=time.time()
    sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        print("try ", id_)
        port_ = find_free_ports(self_addr[1]+1,self_addr[1]+49,  self_addr)
        sock.bind((self_addr[0], port_))
        sock.connect(host_addr)
        sock.send(pickle.dumps(('find_id', id_)))
        sock.settimeout(30)
        data = pickle.loads(sock.recv(1024))
        print(data[0], "[", time.time()-st,"] ----" , id_)
        sock.close()
        return data[0]
    except TimeoutError:
        print("timeout")
    except ConnectionRefusedError:
        print("successor may be dead!!")
        sock.close()
        return 0
    except Exception as error:
        traceback.print_exc()
        print(host_addr)
    sock.close()
    print("STARTQUERY IS DIE!!!!!!!!!!!!!!!!!!!!!")
