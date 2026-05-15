import os
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from KAN_PROSPECT import KAN_PROSPECT_ATC,KAN_PROSPECT_ADR
from val import val
from utils import *


###ATC###
def main():
    model_st = KAN_PROSPECT_ATC.__name__  
    modeling = KAN_PROSPECT_ATC 
    random.seed(42)
    params = dict(
        data_root="data",  
        save_dir="save",  
        dataset="human",  
        lr=0.001,  
        batch_size=512
    )
    
    logger = TrainLogger(params)
 
    train_data = torch.load('PretrainATC.pt')
    test_data = torch.load('PretestATC.pt')

    logger.info(f"Number of train: {len(train_data)}")
    logger.info(f"Number of test: {len(test_data)}")
    
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=True)
    device = torch.device( "cpu")
    model = modeling(n_output=1).to(device)

    epochs = 200
    steps_per_epoch = 20
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    pos_weight = torch.tensor([0.8], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    global_step = 0
    global_epoch = 0
    running_loss = AverageMeter()
    model.train()
    save_model = True

    log_data = []
    train_labels, train_preds = [], []
    test_labels, test_preds = [], []

    for i in range(num_iter):
        
        for data in train_loader:
            
            optimizer.zero_grad()
            global_step += 1   
            data.y = data.y.float()    
            data = data.to(device)
            
            pred = model(data)

            loss = criterion(pred, data.y.unsqueeze(1))

            loss.backward()
            
            optimizer.step()
            running_loss.update(loss.item(), data.y.size(0)) 
            
            if global_step % steps_per_epoch == 0:
                global_epoch += 1
                epoch_loss = running_loss.get_average()
                running_loss.reset()
                
                (test_loss, test_acc , test_pre,test_rec, test_f1,  test_mcc,test_auc, test_aupr,  test_labels_batch, test_preds_batch,fpr_test, tpr_test) = val(model, criterion, test_loader, device,pred_save=False,threshold=0.5,plot_distribution=False)

                test_labels.extend(test_labels_batch)
                test_preds.extend(test_preds_batch)
        
                msg = (
                    f"epoch-{global_epoch}, loss-{epoch_loss:.4f} "
                    f"test_loss-{test_loss:.4f} "
                    f"test_acc-{test_acc:.4f} , "
                    f"test_pre-{test_pre:.4f} , "
                    f"test_rec-{test_rec:.4f} , "
                    f"test_f1-{test_f1:.4f} , "
                    f"test_mcc-{test_mcc:.4f} , "
                    f"test_auc-{test_auc:.4f} , "
                    f"test_aupr-{test_aupr:.4f},"
                    
                )
                print(msg)

                if save_model:
                    save_model_dict(model, logger.get_model_dir(), msg)
                    torch.save(model.state_dict(), "PretrainATC.pth")
                    torch.save(KAN_PROSPECT_ATC(), "EntirePretrainATC.pth")     
       
        if global_epoch >= 90:
            break
    
    df_log = pd.DataFrame(log_data)
    
    df_log.to_csv(os.path.join(params['save_dir'], 'ATC_result.csv'), index=False)

if __name__ == "__main__":
    main() 

###ADR###
def main():
    model_st = KAN_PROSPECT_ADR.__name__ 
    modeling = KAN_PROSPECT_ADR 
    
    params = dict(
        data_root="data", 
        save_dir="save",  
        dataset="human", 
        lr=0.001,  
        batch_size= 1024
    )
    
    logger = TrainLogger(params)
    
    train_data = torch.load('PretrainADR.pt')
    test_data = torch.load('PretestADR.pt')

    
    logger.info(f"Number of train: {len(train_data)}")
    logger.info(f"Number of test: {len(test_data)}")
    
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=True)
    
    device = torch.device("cpu")
    model = modeling(n_output=1).to(device)#2

    epochs = 200
    steps_per_epoch = 10
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    pos_weight = torch.tensor([1.5], device=device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    global_step = 0
    global_epoch = 0
    running_loss = AverageMeter()
    model.train()
    save_model = True

    log_data = []

    for i in range(num_iter):
        for data in train_loader:
            optimizer.zero_grad()
            global_step += 1   

            data.y = data.y.float() 
 
            data = data.to(device)

            pred = model(data)
            loss = criterion(pred, data.y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss.update(loss.item(), data.y.size(0)) 

            if global_step % steps_per_epoch == 0:
                global_epoch += 1
                epoch_loss = running_loss.get_average()
                running_loss.reset()
                
                test_loss, acc, pre, rec, auc,aupr,f1,mcc= val(model, criterion, test_loader, device)
                
                log_entry = {
                    'epoch': global_epoch,
                    'loss': epoch_loss,
                    'test_loss': test_loss,
                }
                log_data.append(log_entry)

                
                msg = "epoch-%d, loss-%.4f, test_loss-%.4f, test_acc-%.4f, test_pre-%.4f, test_rec-%.4f,test_auc-%.4f,test_aupr-%.4f,test_f1-%.4f,test_mcc-%.4f"  % (
                    global_epoch, epoch_loss,test_loss, acc, pre, rec, auc,aupr,f1,mcc)
                
                
                
                print(msg)
             

                if save_model:
                    save_model_dict(model, logger.get_model_dir(), msg)
                    torch.save(model.state_dict(), "PretrainADR.pth")
                    
    df_log = pd.DataFrame(log_data)
    df_log.to_csv(os.path.join(params['save_dir'], 'ATC_result.csv'), index=False)

if __name__ == "__main__":
    main() 