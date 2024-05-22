"""
This is implementation of Federated learning with P2P communication with DHT-chord protocol.
"""
from src.fednode import fednode
from src.utils.utils import handle_args_f

if __name__ == '__main__':
    print("         fedchord - OK")
    this_addr, help_addr, logger, device, config= handle_args_f()

    if config['thisid'] in range(config['splitnum']):
        case = [config['splitnum'],config['thisid']]
    else:
        raise ValueError("case_n must be in range (%d)" %config["splitnum"])
        
    node = fednode(config, logger, (this_addr), (help_addr), device, case)