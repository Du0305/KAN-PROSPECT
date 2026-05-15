import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             precision_recall_curve, auc, roc_curve)
from utils import *

def val(model, criterion, dataloader, device, pred_save=False,threshold=0.0, plot_distribution=True, save_path=None):
    model.eval()
    loss_fn = AverageMeter() 

    pred_list = []
    pred_cls_list = []
    label_list = []

    for data in dataloader:
        data.y = data.y.float() 
        data = data.to(device)  
        with torch.no_grad():  
            pred = model(data) 
            loss = criterion(pred, data.y.unsqueeze(1))
            pred_prob = pred
            pred_cls = (pred_prob > threshold).float()
            pred_list.append(pred_prob.detach().cpu().numpy())
            pred_cls_list.append(pred_cls.detach().cpu().numpy())
            label_list.append(data.y.detach().cpu().numpy())
            loss_fn.update(loss.item(), data.y.size(0))

    pred = np.concatenate(pred_list, axis=0).reshape(-1)
    pred_cls = np.concatenate(pred_cls_list, axis=0).reshape(-1)
    label = np.concatenate(label_list, axis=0).reshape(-1)
    if pred_save:
        df_out = pd.DataFrame({
        'pred_prob': pred,      
        'pred_cls':  pred_cls,   
        'label':     label       
    })
        df_out.to_csv('df_out.csv', index=False)
    
    acc = accuracy_score(label, pred_cls) 
    pre = precision_score(label, pred_cls)
    rec = recall_score(label, pred_cls)
    f1 = f1_score(label, pred_cls)  
    mcc = matthews_corrcoef(label, pred_cls) 
    auc_roc = roc_auc_score(label, pred) 
    fpr_train, tpr_train, _ = roc_curve(label, pred)

    precision, recall, _ = precision_recall_curve(label, pred)
    aupr = auc(recall, precision)
    
    epoch_loss = loss_fn.get_average() 
    loss_fn.reset()

    model.train() 
    
    return epoch_loss,  acc, pre, rec, f1, mcc, auc_roc, aupr, label, pred,fpr_train, tpr_train
