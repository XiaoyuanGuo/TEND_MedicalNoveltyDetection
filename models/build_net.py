import logging
from .MNIST_AE import *
from .CIFAR10_AE import *
from .ConvAutoencoder import *


def build_ae_net(dataset_name, capacity, channel):   
    
    logger = logging.getLogger()
    logger.info("Build embnet for {}".format(dataset_name))
    assert dataset_name in ['mnist', 'cifar10', 'ivc-filter', 'rsna']
    
    embnet = None
    
    if dataset_name == "mnist":
        embnet = MNIST_AE()
        
    elif dataset_name == "cifar10":
        embnet = CIFAR10Autoencoder()
        
    elif dataset_name == "ivc-filter" or dataset_name == "rsna":
        embnet = Autoencoder(capacity, channel)

    return embnet

def build_cls_net(dataset_name, embnet, capacity):

    logger = logging.getLogger()
    logger.info("Build cls_model network for {}".format(dataset_name))

    assert dataset_name in ['mnist', 'cifar10', 'ivc-filter', 'rsna']
    
    cls_model = None
    if dataset_name == "mnist":
        embnet.eval()
        cls_model = mnistClassifier(embnet)
        
    elif dataset_name == "cifar10":
        embnet.eval()
        cls_model = CIFAR10AutoencoderCLS(embnet) 
        
    if dataset_name == "ivc-filter" or dataset_name == "rsna":
        embnet.eval()
        cls_model = aeClassifier(embnet, capacity)
    
    return cls_model
    
