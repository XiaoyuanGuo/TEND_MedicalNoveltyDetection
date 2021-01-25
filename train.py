import os
import time
import copy
import torch
import logging
import numpy as np

def load_ckpt(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model

def save_checkpoint(state, filename):
    """Save checkpoint if a new best is achieved"""
    print ("=> Saving a new best")
    torch.save(state, filename)  # save checkpoint
    

def train_embnet(embnet, recon_loss, ae_optimizer, ae_num_epochs, ae_dataloaders, ae_dataset_sizes, dataset_name, device):
    logger = logging.getLogger()
    since = time.time()
    best_loss = np.inf
    trainloss = []
    valloss = []
    logger.info("---------Stage-1 AE training----------")
    for epoch in range(ae_num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, ae_num_epochs))
        
        for phase in ['train','val']:
            if phase == 'train':
                embnet.train()
            else:
                embnet.eval()
                
            running_loss= 0.0
            
            for idx, inputs in enumerate(ae_dataloaders[phase]):
                ae_optimizer.zero_grad()
                with torch.set_grad_enabled(phase =='train'):
                    images, _ = inputs
                    images = images.to(device) 
                    recon_imgs = embnet(images)
                    
                    loss = recon_loss(recon_imgs, images)
                       
                    if phase == 'train':
                        loss.backward()
                        ae_optimizer.step()
                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / ae_dataset_sizes[phase]
            
            logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'train':
                trainloss.append(epoch_loss)
            else:
                valloss.append(epoch_loss)
                
            if not os.path.exists('./weights/'):
                os.mkdir('./weights/')
            if not os.path.exists('./weights/'+dataset_name):
                os.mkdir('./weights/'+dataset_name)
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(embnet.state_dict())
                save_checkpoint(state={'epoch': epoch, 
                                      'model_state_dict': embnet.state_dict(),
                                      'best_loss':best_loss,
                                      'optimizer_state_dict': ae_optimizer.state_dict()},
                                filename = './weights/'+dataset_name+"/ae.pt")
                print()
    time_elapsed = time.time()-since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    logger.info('Best val loss:{:4f}'.format(best_loss))
            
    embnet.load_state_dict(best_model_wts)
           
    return embnet


def init_center_c(train_loader, dataset_name, net, device, eps=0.1,):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    
    if dataset_name == "mnist":
        tnum = 32
    elif dataset_name == "cifar10":
        tnum = 96*2*2
    elif dataset_name == "ivc-filter" or dataset_name == "rsna":
        tnum = 512
    else:
        tnum = 0
    c = torch.zeros(tnum, device=device)

    net.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(train_loader):
            # get the inputs of the batch
            data, _ = inputs
            org_imgs, _ = data
            org_imgs = org_imgs.to(device)
            outputs = net.get_embedding(org_imgs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c

def train_cls_model(cls_model, cls_loss, cls_optimizer, cls_num_epochs, warmup_epochs, cls_dataloaders, cls_dataset_sizes, dataset_name, device, R=150, c=None):    
    assert dataset_name in ['mnist', 'cifar10', 'rsna', 'ivc-filter']
    logger = logging.getLogger()
    logger.info("---------Stage-2 TEND training----------")
    since = time.time()
    best_loss = np.inf
    best_fpr = np.inf
    trainloss = []
    valloss = []

    for epoch in range(cls_num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, cls_num_epochs))
        for phase in ['train','val']:
            if phase == 'train':
                cls_model.train()
                if epoch == warmup_epochs and c == None:
                    c = init_center_c(cls_dataloaders[phase], dataset_name, cls_model, device)
            else:
                cls_model.eval()
            running_loss= 0.0
                       
            for idx, inputs in enumerate(cls_dataloaders[phase]):
                
                cls_optimizer.zero_grad()
                with torch.set_grad_enabled(phase =='train'):
                    if phase == 'train':
                        data, targets = inputs
                        org_imgs, tfm_imgs = data
                        org_imgs, tfm_imgs = org_imgs.to(device), tfm_imgs.to(device)
                    
                        org_targets, tfm_targets = targets
                        org_targets, tfm_targets = org_targets.to(device), tfm_targets.to(device)
                    
                        all_imgs = torch.cat([org_imgs, tfm_imgs], dim=0)
                        all_targets = torch.cat([org_targets, tfm_targets], dim=0)
                        all_targets = torch.unsqueeze(all_targets, dim=1).float()
                        
                        preds = cls_model(all_imgs)
                        loss = cls_loss(preds, all_targets) 
                            
                        if isinstance(cls_model,torch.nn.DataParallel):
                            cls_model = cls_model.module

                        if epoch >= warmup_epochs:
                            outputs = cls_model.get_embedding(org_imgs)
                            dist = torch.sum((outputs - c) ** 2, dim=1)
                            loss += torch.mean(dist)
                    
                            toutputs = cls_model.get_embedding(tfm_imgs)
                            tdist = torch.sum((toutputs - c) ** 2, dim=1)
                            loss += torch.mean(torch.nn.functional.relu(R-tdist))
                
                        loss.backward()
                        cls_optimizer.step()
                    else:
                        data, targets = inputs
                        all_imgs = data.to(device)
                        all_targets = targets.to(device)
                        all_targets = torch.unsqueeze(all_targets, dim=1).float()
                        
                        preds = cls_model(all_imgs)
                        loss = cls_loss(preds, all_targets) 
                        if isinstance(cls_model,torch.nn.DataParallel):
                            cls_model = cls_model.module
                        if epoch >= warmup_epochs:
                            nrm_indices = torch.nonzero((targets == 0))
                            nrm_imgs = torch.index_select(data, 0, nrm_indices.squeeze(1))
                            
                            ood_indices = torch.nonzero((targets == 1))
                            ood_imgs = torch.index_select(data, 0, ood_indices.squeeze(1))
                            
                            nrm_imgs, ood_imgs = nrm_imgs.to(device), ood_imgs.to(device)
                            
                            if nrm_indices.shape[0] != 0:
                                nrm_outputs = cls_model.get_embedding(nrm_imgs)
                                nrm_dist = torch.sum((nrm_outputs - c) ** 2, dim=1)
                                loss += torch.mean(nrm_dist)
                    
                            if ood_indices.shape[0] != 0:
                                ood_outputs = cls_model.get_embedding(ood_imgs)
                                ood_dist = torch.sum((ood_outputs - c) ** 2, dim=1)
                                loss += torch.mean(torch.nn.functional.relu(R-tdist))
                        
                running_loss += loss.item() * all_imgs.shape[0]
            epoch_loss = running_loss / cls_dataset_sizes[phase]
            
            logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'train':
                trainloss.append(epoch_loss)
            elif phase == 'val' and epoch >= warmup_epochs: 
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_cls_model_wts = copy.deepcopy(cls_model.state_dict())
                    save_checkpoint(state={'epoch': epoch, 
                                      'model_state_dict': cls_model.state_dict(),
                                      'best_loss':best_loss,
                                      'optimizer_state_dict': cls_optimizer.state_dict()},
                                filename = './weights/'+dataset_name+"/tend.pt")
                    print()
    time_elapsed = time.time()-since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    logger.info('Best val loss:{:4f}'.format(best_loss))
            
    cls_model.load_state_dict(best_cls_model_wts)
    return cls_model, c
