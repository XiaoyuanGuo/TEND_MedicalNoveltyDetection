import os
import time
import copy
import click
import torch
import logging
import numpy as np
from torch import nn
from pathlib import Path

from utils.config import Config
from utils.tsne import get_tsne
from utils.plot_roc import plot_roc
from datasets.build_dataset import *
from models.build_net import *
from train import train_embnet, train_cls_model, load_ckpt
from evaluate import ae_evaluate, tend_evaluate



################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', 'rsna', 'ivc-filter']))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--capacity', type=int, default=16, help='Specify Convoluation layer channel unit')
@click.option('--channel', type=int, default=1, help='Specify image channel')
@click.option('--ae_batch_size', type=int, default=32, help='Batch size for mini-batch training.')
@click.option('--ae_n_epochs', type=int, default=50, help='Stage-1 autoencoder training epochs')
@click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for autoencoder network training.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder network training. Default=0.001')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--load_ae_model', type=bool, default=False, help='Whether load previous trained model')
@click.option('--ae_model_path', type=click.Path(exists=True), default="./weights/", help='Model file path')
@click.option('--cls_batch_size', type=int, default=32, help='Batch size for TEND training.')
@click.option('--cls_n_epochs', type=int, default=50, help='Stage-2 tend training epochs')
@click.option('--warmup_epochs', type=int, default=10, help='Stage-2 tend warmup training epochs')
@click.option('--cls_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for tend network training.')
@click.option('--cls_lr', type=float, default=0.001,
              help='Initial learning rate for tend network training. Default=0.001')
@click.option('--cls_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for tend.')
@click.option('--r', type=float, default=150.0, help='Margin radius.')
@click.option('--load_cls_model', type=bool, default=False, help='Whether load previous trained model')
@click.option('--cls_model_path', type=click.Path(exists=True), default="./weights/", help='Model file path')
@click.option('--normal_class', type=list, default=[0],
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--roc_visualization', type=bool, default=False, help='Whether visualize the roc curve for AE & Tend')
@click.option('--tsne_visualization', type=bool, default=False, help='Whether visualize the tsne embedding for AE & Tend')
@click.option('--load_config', type=bool, default=False, help='Whether use previous log')
@click.option('--config_path', type=click.Path(exists=True), default="./logs/",
              help='Config JSON-file path (default: None).')



def main(dataset_name, data_path, capacity, channel, ae_batch_size, ae_n_epochs, ae_optimizer_name, ae_lr, ae_weight_decay, load_ae_model, ae_model_path, cls_batch_size, cls_n_epochs, warmup_epochs, cls_optimizer_name, cls_lr, cls_weight_decay, r , load_cls_model, cls_model_path, normal_class, roc_visualization, tsne_visualization, load_config, config_path):
    
    # Get configuration
    cfg = Config(locals().copy())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0      
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = "./logs/"+ dataset_name +'/log.txt'
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    if not os.path.exists("./logs/"+ dataset_name):
        os.mkdir("./logs/"+ dataset_name)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if load_config:
        cfg.load_config(import_json=config_path)
        logger.info('Loaded configuration from %s.' % config_path)
    
    ae_model_path = cfg.settings['ae_model_path']+dataset_name+"/ae.pt"
    cls_model_path = cfg.settings['cls_model_path']+dataset_name+"/tend.pt"
    config_path = cfg.settings['config_path']+dataset_name+"/config.json"
    
    # Print arguments
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % cfg.settings['data_path'])
    logger.info('Dataset: %s' % cfg.settings['dataset_name'])
    logger.info('AE Conv capacity: %d' % cfg.settings['capacity'])
    logger.info('image channel: %d' % cfg.settings['channel'])
    logger.info('------------Stage-1--------------')
    logger.info('AE batchsize: %d' % cfg.settings['ae_batch_size'])
    logger.info('AE epochs: %d' % cfg.settings['ae_n_epochs'])
    logger.info('AE optimizer: %s' % cfg.settings['ae_optimizer_name'])
    logger.info('AE lr: %f' % cfg.settings['ae_lr'])
    logger.info('AE weight_decay: %f' % cfg.settings['ae_weight_decay'])
    logger.info('AE load_model: %r' % cfg.settings['load_ae_model'])
    logger.info('AE model_path %s' % ae_model_path)
    logger.info('------------Stage-2--------------')
    logger.info('TEND batchsize: %d' % cfg.settings['cls_batch_size'])
    logger.info('TEND epochs: %d' % cfg.settings['cls_n_epochs'])
    logger.info('TEND warmup_epochs: %d' % cfg.settings['warmup_epochs'])
    logger.info('TEND optimizer: %s' % cfg.settings['cls_optimizer_name'])
    logger.info('TEND lr: %f' % cfg.settings['cls_lr'])
    logger.info('TEND weight_decay: %f' % cfg.settings['cls_weight_decay'])
    logger.info('TEND margin radius: %f' % cfg.settings['r'])
    logger.info('TEND roc_visualization: %r'% cfg.settings['roc_visualization'])
    logger.info('TEND tsne_visualization: %r'% cfg.settings['tsne_visualization'])
    logger.info('TEND load_model: %r' %  cfg.settings['load_cls_model'])
    logger.info('TEND model_path: %s' % cls_model_path)
    
    
    nrm_cls_info = ""
    for i in range(0, len(cfg.settings['normal_class'])):
        nrm_cls_info = nrm_cls_info + str(cfg.settings['normal_class'][i])
    logger.info('Normal class:' + nrm_cls_info)
    
    logger.info("AutoEncoder ==> embnet")
    logger.info("TEND ==> cls_model")

    ae_dataloaders, ae_dataset_sizes = build_ae_dataset(cfg.settings['dataset_name'], cfg.settings['data_path'], cfg.settings['ae_batch_size'], cfg.settings['normal_class'])
    
    #build embnet based on dataeset_name
    embnet = build_ae_net(cfg.settings['dataset_name'], cfg.settings['capacity'], cfg.settings['channel'])    
    embnet = embnet.to(device)
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        embnet = nn.DataParallel(embnet)
                
    recon_loss = nn.MSELoss()
    amsgrad = False
    if cfg.settings['ae_optimizer_name'] == "amsgrad":
        amsgrad=True
    ae_optimizer = torch.optim.Adam(embnet.parameters(), lr=cfg.settings['ae_lr'], weight_decay=cfg.settings['ae_weight_decay'], amsgrad=amsgrad)
    
    if load_ae_model: # load pretrained model
        logger.info("---------AE load trained model--------")
        embnet = load_ckpt(ae_model_path, embnet, ae_optimizer)
        
    embnet = train_embnet(embnet, recon_loss, ae_optimizer, cfg.settings['ae_n_epochs'], ae_dataloaders, ae_dataset_sizes, cfg.settings['dataset_name'], device)
    embnet.eval()
    
    aeEmbs, aeTargets, ae_fpr, ae_tpr = ae_evaluate(embnet, recon_loss, ae_dataloaders["test"], device)

    #start training TEND
    if isinstance(embnet,torch.nn.DataParallel):
        embnet = embnet.module   
    cls_model = build_cls_net(cfg.settings['dataset_name'], embnet, cfg.settings['capacity']) 
    cls_model = cls_model.to(device)
    
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        cls_model = nn.DataParallel(cls_model)
    
    cls_dataloaders, cls_dataset_sizes = build_tend_dateset(cfg.settings['dataset_name'], cfg.settings['data_path'], cfg.settings['cls_batch_size'], cfg.settings['normal_class']) 
    cls_loss = nn.BCELoss()
    amsgrad = False
    if cfg.settings['cls_optimizer_name'] == "amsgrad":
        amsgrad=True
    cls_optimizer = torch.optim.Adam(cls_model.parameters(), lr=cfg.settings['cls_lr'], weight_decay=cfg.settings['cls_weight_decay'], amsgrad=amsgrad)
    
    if load_cls_model: # load pretrained model
        logger.info("---------TEND load trained model----------")
        if isinstance(cls_model,torch.nn.DataParallel):
            cls_model = cls_model.module
        cls_model = load_ckpt(cls_model_path, cls_model, cls_optimizer)
            
    cls_model, c = train_cls_model(cls_model, cls_loss, cls_optimizer, cfg.settings['cls_n_epochs'], cfg.settings['warmup_epochs'], cls_dataloaders, cls_dataset_sizes, cfg.settings['dataset_name'], device, cfg.settings['r'])
    cls_model.eval()
    clsEmbs, clsTargets, cls_fpr, cls_tpr = tend_evaluate(cls_model, cls_loss, cls_dataloaders["val"], c, device)
    cfg.save_config(export_json=config_path)
    
    if cfg.settings['roc_visualization']:
        plot_roc([ae_fpr, cls_fpr], [ae_tpr, cls_tpr], cfg.settings['dataset_name'])
        
    if cfg.settings['tsne_visualization']:
        print("Will visualize the embedding of AE and TEND")
        get_tsne(aeEmbs, aeTargets, cfg.settings['dataset_name'], stage=1)
        get_tsne(clsEmbs, clsTargets, cfg.settings['dataset_name'], stage=2)
            

if __name__ == '__main__':
    main()
