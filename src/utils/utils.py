import textwrap
import logging
import hashlib
import subprocess
import socket
import logging
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

__all__ = ['contain', 'MultiLineFormatter', 'hash', 'get_global_ip', 'get_self_ip', 'find_free_ports','handle_args', 'handle_args_f']

def contain(id, begin, end):
    """
    check if id is between begin and end
    """
    if begin < end:
        return begin < id <= end
    elif begin > end:
        return begin < id or id <= end
    return False

class MultiLineFormatter(logging.Formatter):
    def format(self, record):
        message = record.msg
        record.msg = ''
        header = super().format(record)
        msg = textwrap.indent(message, ' ' * len(header)).lstrip()
        record.msg = message
        return header + msg

def handle_args():
    this_ip=get_self_ip()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port","-p", help="this peer's port number", type=int, default=12000)
    parser.add_argument("--addr","-a", help="this peer's ip address", type=str, default=this_ip)
    parser.add_argument("--help_port","-P", help="help peer's port number", type=int, default=-1)
    parser.add_argument("--help_addr","-A", help="help peer's ip address", type=str, default=this_ip)
    parser.add_argument("--log", help="enable log", action="store_true", default=False)
    parser.add_argument("--debug", help="enable log(debug)", action="store_true", default=False)
    parser.add_argument("--container","-c", help="container", action="store_true", default=False)
    args = parser.parse_args()

    this_addr = (args.addr, args.port)
    if args.help_port == -1:
        help_addr = None
    else:
        help_addr = (args.help_addr, args.help_port)

    formatter = MultiLineFormatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
    )

    if args.log == True:
        logger = logging.getLogger()
        #log_handler= logging.FileHandler("logs/%s.log" %(str(this_addr[1])), mode='w', encoding=None, delay=False)
        log_handler= logging.StreamHandler()
        log_handler.setFormatter(formatter)
        if args.debug==True:
            log_handler.setLevel(logging.DEBUG)
        else:
            log_handler.setLevel(logging.INFO)
        logger.addHandler(log_handler)

        con_handler= logging.StreamHandler()
        con_handler.setFormatter(formatter)
        con_handler.setLevel(logging.CRITICAL)
        logger.addHandler(con_handler)

        logger.setLevel(logging.DEBUG)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        log_handler= logging.StreamHandler()
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)



    return this_addr, help_addr, logger, args.container

def handle_args_f():
    this_ip=get_self_ip()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port","-p", help="this peer's port number", type=int, default=12000)
    parser.add_argument("--addr","-a", help="this peer's ip address", type=str, default=this_ip)
    parser.add_argument("--help_port","-P", help="help peer's port number", type=int, default=-1)
    parser.add_argument("--help_addr","-A", help="help peer's ip address", type=str, default=this_ip)
    parser.add_argument("--log", help="enable log", action="store_true", default=False)
    parser.add_argument("--debug", help="enable log(debug)", action="store_true", default=False)
    parser.add_argument("--container","-c", help="if in container", action="store_true", default=False)
    parser.add_argument("--thisid", "-t", help="id of dataset partition", type=int, default=0)
    parser.add_argument("--splitnum", "-s", help="total number of peer(= number of dataset partition)", type=int, default=0)
    #add for federated learning
    parser.add_argument("--gpu","-g", help="GPU num (-1 if use cpu)", type=int, default=0)
    parser.add_argument("--iid","-i", help="for use custom dataset", action="store_false", default=True)
    parser.add_argument("--model","-m",  help="model's name", type=str, choices=["resnet18", "cnn", "lstm"], default="resnet18")
    parser.add_argument("--dataset","-d", help="dataset's name", type=str, default="cifar10")
    parser.add_argument("--slow", help="the number of slow node (straggler)", type=int, default=0)
    parser.add_argument("--dfedavgm", help="run DfedAvgM mode", action="store_true", default=False)
    args = parser.parse_args()

    this_addr = (args.addr, args.port)
    if args.help_port == -1:
        help_addr = None
    else:
        help_addr = (args.help_addr, args.help_port)

    formatter = MultiLineFormatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
    )

    if args.log == True:
        logger = logging.getLogger()
        log_handler= logging.FileHandler("logs/%s.log" %(str(this_addr[1])), mode='w', encoding=None, delay=False)
        log_handler.setFormatter(formatter)
        if args.debug==True:
            log_handler.setLevel(logging.DEBUG)
        else:
            log_handler.setLevel(logging.INFO)
        logger.addHandler(log_handler)

        con_handler= logging.StreamHandler()
        con_handler.setFormatter(formatter)
        con_handler.setLevel(logging.CRITICAL)
        logger.addHandler(con_handler)

        logger.setLevel(logging.DEBUG)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        log_handler= logging.StreamHandler()
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)

    #add for federated learning
    if args.gpu == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(args.gpu)

    config={"this_addr":this_addr, "help_addr":help_addr, "logger": logger, "device": device}
    return this_addr, help_addr, logger, device, vars(args)

def hash(addr, NUM_OF_BITS=6):
    """
    hash function, that generated id of node by address, using sha1
    """
    return int(hashlib.sha1(str(addr).encode()).hexdigest(), 16) % (2**NUM_OF_BITS)


def get_global_ip():
    """
    get global ip address
    """
    return subprocess.check_output("wget http://ipecho.net/plain -O - -q ; echo", shell=True).decode().strip()

def get_self_ip():
    return socket.gethostbyname(socket.gethostname())

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