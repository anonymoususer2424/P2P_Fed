from .resnet import *
from .mobilenetv2 import *
from .cnn import *
from .rnn import *
getmodel={ # just for now 
    'ComplexCNN':ComplexCNN,
    'SimpleCNN':SimpleCNN,
    'ResNet18':ResNet18,
    'RNN_Shakespeare':RNN_Shakespeare_80
    
}