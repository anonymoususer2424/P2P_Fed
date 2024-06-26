import socket, time
import threading
import hashlib
import pickle
import selectors
import traceback
import copy
import numpy as np
from .utils.utils import contain, find_free_ports, get_self_ip, handle_args
NUM_OF_BITS = 32

__all__ = ['P2PNode']

#TODO : error handling, define solo remain node's behavior

class P2PNode:
    """
    node of Chord DHT
    node's id is generated by hash function
    """
    
    def __init__(self, logger, addr, host_addr=None, container=False, data_con=[1,0]):
        print("         node - OK")
        self.addr = addr
        if container:
            self.realaddr = (get_self_ip(), self.addr[1])
        else:
            self.realaddr = addr
        
        self.host_addr = host_addr
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.realaddr)
        self.socket.listen()

        node_positions = np.linspace(0,2**32,data_con[0],dtype=np.int64,endpoint=False) # For real case this will be return valure of __hash(self.addr).

        self.id: int
        self.id=node_positions[data_con[1]]
        self.checking=1 #for container..
        
        self.finger_table = list(("",0) for i in range(NUM_OF_BITS))
        self.__init_finger_table()
        self.predecessor_addr = None
        self.predecessor_id = -1
        self.successor_addr = self.addr
        self.successor_id = self.id
        self.socketlist=[self.socket]
        self.alive = True
        self.stable_time = 1
        """
        loop for stablize after few seconds, to update finger table, successor and predecessor after join or unjoin of other nodes
        """
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.socket, selectors.EVENT_READ, self.__accept_handler)
        self.listen_t = threading.Thread(target=self.run, daemon=True, name="run")
        
        
        self.listen_t.start()
        print("         join - ST")
        while 1:
            try:
                self.join()
                break
            except(ConnectionError):
                print("Connection Error start again..")
                time.sleep(1)
                pass
        print("         join - OK")
        self.daemon_t = threading.Thread(target=self.do_daemon, daemon=True, name="daemon")
        self.daemon_t.start() 
        
        self.mainjob()
    
    ######################## methods called by init #######################
    
    def mainjob(self):

        while self.alive:
            time.sleep(5)
        #pass
            
    def __init_finger_table(self):
        """
        initialize finger table
        """
        for i in range(NUM_OF_BITS):
            self.finger_table[i] = (self.addr, self.id)    
    
    def __hash(self, addr):
        """
        hash function, that generated id of node by address, using sha1
        """
        return int(hashlib.sha1(str(addr).encode()).hexdigest(), 16) % (2**NUM_OF_BITS)
    
    ##################### daemon thread called by init ####################
    
    def do_daemon(self):
        i=0
        while self.checking:
            print("         Chording - ",i)
            self.__stabilize()
            self.__fix_fingers()
            time.sleep(self.stable_time)
            i+=1
    

    def __stabilize(self):
        """
        stabilize the network
        """
        if(self.successor_addr != self.addr):
            try:
                sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                port_ =find_free_ports(self.realaddr[1]+1,self.realaddr[1]+49,  self.realaddr)
                sock.bind((self.realaddr[0], port_))

                
                sock.connect(self.successor_addr)
                sock.send(pickle.dumps(('get_predecessor', self.addr, self.id)))

                data = self.recv_data(sock)

                if data[1] != -1 and contain(data[1], self.id, self.successor_id):
                    self.successor_addr = data[0]
                    self.successor_id = data[1]
                self._notify()

            except:
                """
                connect to next address in finger table
                """
                message = "Node {} left".format(self.successor_addr)
                print(message)
                for i,addr in enumerate(self.finger_table):
                    if (addr[0] != self.successor_addr):
                        self.successor_addr = addr[0]
                        self.successor_id = addr[1]
                        for j in range(i):
                            self.finger_table[j] = self.finger_table[i]

                        break
                    else:
                        self.finger_table[i]= (self.predecessor_addr,self.predecessor_id)
                if(self._notify_leave()==False):

                    self.successor_addr = self.predecessor_addr
                    self.successor_id = self.predecessor_id

            if sock:    #if connect remain, close here.
                sock.close
                   
    def __fix_fingers(self):
        """
        fix fingers
        """
        if self.successor_addr != self.addr:
            self._update_finger_table()
            #self.print_finger_table()
 
    ################### listening thread called by init ###################
    
    def run(self):
        """
        thread for listening
        """
        while self.alive:
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.socket, selectors.EVENT_READ, self.__accept_handler)
            while self.alive:
                for (key,mask) in self.selector.select():
                    key: selectors.SelectorKey
                    srv_sock, callback = key.fileobj, key.data
                    callback(srv_sock, self.selector)

    def __accept_handler(self, sock: socket.socket, sel: selectors.BaseSelector):
        """
        accept connection from other nodes
        """
        conn: socket.socket
        conn, addr = sock.accept()
        sel.register(conn, selectors.EVENT_READ, self.__read_handler)
        

    def __read_handler(self, conn: socket.socket, sel: selectors.BaseSelector):
        """
        read data from other nodes
        """
        try:
            data = conn.recv(1024)
            data = pickle.loads(data)
            tr=threading.Thread(target=self._handle, args=((data,conn)), daemon=True)
            tr.start()
            sel.unregister(conn)

        except Exception as e:
            print("##ERROR IN READ HANDER!!")

    ################## core method thread called by init ##################

    def join(self):
        """
        join to the network
        """
        if self.host_addr:
            self._find_successor(self.host_addr)
            self._notify()
        else:
            self.successor_addr = self.addr
            self.predecessor_addr = self.addr
            self.successor_id = self.id
            self.predecessor_id = self.id
    
    def _find_successor(self, addr):
        """
        find successor of id, data sent by pickle
        """
        T=0
        data=("",0)
        while True:
            try:
                sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                port_ =find_free_ports(self.realaddr[1]+1,self.realaddr[1]+49,  self.realaddr)
                sock.bind((self.realaddr[0], port_))
                sock.connect(addr)
                break
            except socket.timeout:
                print("Can't connected in 5 sec. Try again(%d)" %(T))
                T+=1
                pass
        sock.send(pickle.dumps(('find_successor', self.addr, self.id)))
        try:
            data=self.recv_data(sock)

        except socket.timeout:
                print("Can't recv in 30 sec. It may dead me:(%s) to:(%s)", str(self.addr), str(addr))
        """
        get successor address and id with mutex
        """
        self.successor_addr = data[0]
        self.successor_id = data[1]
        sock.close()
  
    def _notify(self):
        """
        notify successor about self
        """
        sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_ =find_free_ports(self.realaddr[1]+1,self.realaddr[1]+49,  self.realaddr)
        sock.bind((self.realaddr[0], port_))

        sock.connect(self.successor_addr)
        sock.send(pickle.dumps(('notify', self.addr, self.id)))
        print("             >")
        sock.close()

    def _update_finger_table(self):
        """
        update finger table
        """
        self.finger_table[0] = (self.successor_addr, self.successor_id)
        for i in range(1, NUM_OF_BITS):
            id = (self.id + 2 ** i) % 2 ** NUM_OF_BITS
            addr = self._find_successor_by_id(id, None)
            if(addr!=None):
                if self.finger_table[i] != addr:
                    self.finger_table[i] = addr

    def _find_successor_by_id(self, id, conn: socket.socket):
        """
        find successor of id
        """
        message= "find_successor_by_id {}, {}".format(id, threading.current_thread().name)
        if(id==self.id):
            return((self.addr, self.id))
        
        elif contain(id, self.id, self.successor_id):
            if conn:
                conn.send(pickle.dumps((self.successor_addr, self.successor_id)))
                conn.close()
            else:
                return (self.successor_addr, self.successor_id)
        else: 
            if conn:
                self._find_closest_preceding_finger(id, conn)
            else:
                return self._find_closest_preceding_finger(id, None)
    
    def _find_closest_preceding_finger(self, id:int, conn: socket.socket):
        """
        find closest preceding finger
        """
        check=False
        for i in range(NUM_OF_BITS - 1, -1, -1):
            
            if contain(self.finger_table[i][1], self.id, id):
                check=True
                try:
                    sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    port_ =find_free_ports(self.realaddr[1]+1,self.realaddr[1]+49,  self.realaddr)
                    sock.bind((self.realaddr[0], port_))

                    sock.connect(self.finger_table[i][0])
                    sock.send(pickle.dumps(('find_successor_by_id', self.addr, self.id, id)))
                    data=self.recv_data(sock)
                    sock.close()
                except:
                    data = self.finger_table[(i+1)%NUM_OF_BITS]
                    
                if conn:
                    conn.send(pickle.dumps(data))
                    conn.close()
                else:
                    return data
                break
        if check==False:
            return ((self.successor_addr, self.successor_id))

    def _notify_leave(self):
        """
        notify successor that node's predecessor has left
        """
        try:
            sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port_ =find_free_ports(self.realaddr[1]+1,self.realaddr[1]+49,  self.realaddr)
            sock.bind((self.realaddr[0], port_))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.connect(self.successor_addr)
            sock.send(pickle.dumps(('notify_leave', self.addr, self.id)))
            sock.close()
            print(              "send notify leave to", self.successor_addr)
            return True
        except:
            return False
    
    ######################## for send/recv data ##########################
    
    def send_data(self,addr,data):
        """
        send data to other nodes
        """
        sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_ =find_free_ports(self.realaddr[1]+1,self.realaddr[1]+49,  self.realaddr)
        sock.bind((self.realaddr[0], port_))

        sock.connect(addr)
        sock.send(pickle.dumps(data))
        sock.close()
    
    def recv_data(self,sock):
        """
        receive data from other nodes
        """
        data = pickle.loads(sock.recv(1024))
        sock.close()
        return data

    ######################## for handle a request ########################
    
    def _handle(self, data, conn: socket.socket):
        """
        handle data from other nodes
        """
        if data[0] == 'find_successor':
            self._handle_find_successor(data, conn)
        elif data[0] == 'notify':
            self._handle_notify(data)
        elif data[0] == 'notify_leave':
            self._handle_notify_leave(data)
        elif data[0] == 'get_predecessor':
            self._handle_get_predecessor(data, conn)
        elif data[0] == 'find_successor_by_id':
            self._handle_find_successor_by_id(data, conn)
        elif data[0] == 'find_id':
            data=self._find_closest_preceding_finger(data[1], None)
            conn.send(pickle.dumps(data))
        
        if conn:    #if connect remain, close here.
            conn.close
       
    def _handle_find_successor(self, data, conn):
        """
        handle find_successor request
        """
        if self.id == self.successor_id:
            conn.send(pickle.dumps((self.addr, self.id)))
            conn.close()

        elif contain(data[2], self.id, self.successor_id):
            conn.send(pickle.dumps((self.successor_addr, self.successor_id)))
            conn.close()
        
        else:
            self._find_closest_preceding_finger(data[2], conn)
            return
        
        self.successor_addr = data[1]
        self.successor_id = data[2]
        self.finger_table[0] = (self.successor_addr, self.successor_id)
     
    def _handle_notify(self, data):
        """
        handle notify request
        """
        print("             <")
        if self.predecessor_id == -1 or contain(data[2], self.predecessor_id, self.id) or self.predecessor_id == self.id:
            self.predecessor_addr = data[1]
            self.predecessor_id = data[2]
 
    def _handle_notify_leave(self, data):
        """
        handle notify_leave request
        """
        print(              "predecessor is gone!")
        self.predecessor_addr = data[1]
        self.predecessor_id = data[2]
    
    def _handle_get_predecessor(self, data, conn):
        """
        handle get_predecessor request
        """
        conn.send(pickle.dumps((self.predecessor_addr, self.predecessor_id)))
        conn.close()

    def _handle_find_successor_by_id(self, data, conn: socket.socket):
        """
        handle find_successor_by_id request
        """
        self._find_successor_by_id(data[3], conn)

    # ############################# for logging ############################
    def print_finger_table(self):
        """
        print finger table with ideal id
        """
        finger_table = copy.deepcopy(self.finger_table) # first barrier
        
        message="\n<finger table>"
        try:
            for i,elem in enumerate(finger_table):
                id = (self.id + 2 ** i) % 2 ** NUM_OF_BITS
                message+="\n%d | %s:%d (%d)" % (id, elem[0][0], elem[0][1], elem[1])
            message+="\n"
        except: # second barrier
            print("error in fingertable", self.fingertable, finger_table)
            return

def startquery(id_,host_addr,self_addr):
    st=time.time()
    for i in range(5):
        try:
            print("try ",i)
            sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port_ =find_free_ports(self_addr[1]+1,self_addr[1]+49,  self_addr)
            sock.bind((self_addr[0], port_))
            sock.connect(host_addr)
            sock.send(pickle.dumps(('find_id', id_)))
            sock.settimeout(30)
            data = pickle.loads(sock.recv(1024))
            print(data[0], "[", time.time()-st,"] ----" , id_)
            return data[0]
        except TimeoutError:
            print("timeout")
        except Exception as error:
            traceback.print_exc()
        sock.close()
    print("STARTQUERY IS DIE!!!!!!!!!!!!!!!!!!!!!")


if __name__ == '__main__':
    this_addr, help_addr, logger, container= handle_args()

        
    node = P2PNode(logger, (this_addr), (help_addr))
