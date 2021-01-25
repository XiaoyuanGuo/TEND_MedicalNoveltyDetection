import torch
import logging
import numpy as np
from numpy import sqrt, argmax
from sklearn.metrics import roc_curve
from sklearn.metrics import auc, roc_auc_score


def get_fpr_tpr_auc(Y_label, Y_preds):
    
    fpr, tpr, thresholds = roc_curve(Y_label, Y_preds)
    aucscore = auc(fpr, tpr)
    gmeans = sqrt(tpr * (1-fpr))
    ix = argmax(gmeans)
    logger = logging.getLogger()
    logger.info('Best Threshold=%f, G-Mean=%.3f, FPR=%.3f, TPR=%.3f, AUC=%.3f' % (thresholds[ix], gmeans[ix], fpr[ix], tpr[ix], aucscore))

    return fpr, tpr


def ae_evaluate(embnet, recon_loss, test_dataloader, device):
    logger = logging.getLogger()
    logger.info("---------Stage-1 AE evaluating----------")

    aeEmbs = []
    Targets = []
    anomaly_score = []
    for idx, inputs in enumerate(test_dataloader):
        
        images, targets = inputs
        images = images.to(device) 
        recon_imgs = embnet(images)

        for i in range(0, images.shape[0]):
            loss = recon_loss(recon_imgs[i].unsqueeze(0), images[i].unsqueeze(0))
            anomaly_score.append(loss.item())
            Targets.append(targets[i].detach().cpu().numpy()) 
    
        if isinstance(embnet,torch.nn.DataParallel):
            embnet = embnet.module   
        aeEmbs.append(embnet.get_embedding(images).detach().cpu().numpy())
           
    Y_label = np.array(np.vstack(Targets).squeeze(1),dtype=int).tolist() 
    Y_preds = []
    for s in anomaly_score:
        Y_preds.append((s-np.min(np.array(anomaly_score)))/(np.max(np.array(anomaly_score))-np.min(np.array(anomaly_score))))

    fpr, tpr = get_fpr_tpr_auc(Y_label, Y_preds) 
    
    aeEmbs = np.array(np.vstack(aeEmbs),dtype=int)
    aeEmbs = aeEmbs.reshape(aeEmbs.shape[0],-1)
    Targets = np.array(np.vstack(Targets),dtype=int)
    
    return aeEmbs, Targets, fpr, tpr

    
def tend_evaluate(cls_model, cls_loss, test_dataloader, c, device):
    logger = logging.getLogger()
    logger.info("---------Stage-2 TEND evaluating----------")
    
    clsEmbs = []
    Ypreds = []
    Ylabel = []
    Dist = []
    
    for idx, inputs in enumerate(test_dataloader):
        images, targets = inputs
        all_imgs = images.to(device)
        all_targets = targets.to(device)
        all_targets = torch.unsqueeze(all_targets, dim=1).float()

        ypreds = cls_model(all_imgs)
       
        if isinstance(cls_model,torch.nn.DataParallel):
            cls_model = cls_model.module
            
        clsEmbs.append(cls_model.get_embedding(all_imgs).detach().cpu().numpy())
        for i in range(0, all_imgs.shape[0]):
            outputs = cls_model.get_embedding(all_imgs[i].unsqueeze(0))
            dist = torch.sum((outputs - c) ** 2, dim=1)
            Dist.append(dist.detach().cpu().item())

        Ypreds.append(ypreds.detach().cpu().numpy())
        Ylabel.append(all_targets.detach().cpu().numpy())
    
    Y_label = np.array(np.vstack(Ylabel).squeeze(1),dtype=int).tolist()
    Y_preds = np.array(np.vstack(Ypreds).squeeze(1)).tolist()
    
    lamda = 0.1
    
    newdist = []
    for s in Dist:
        newdist.append((s-np.min(np.array(Dist)))/(np.max(np.array(Dist))-np.min(np.array(Dist))))
    
    newscore = []
    for i in range(0, len(newdist)):
        newscore.append(lamda*newdist[i]+(1-lamda)*Y_preds[i])
    fpr, tpr = get_fpr_tpr_auc(Y_label, newscore)  
    
    clsEmbs = np.array(np.vstack(clsEmbs),dtype=int)
    return clsEmbs, Y_label, fpr, tpr
