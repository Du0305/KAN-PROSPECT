"""
KAN_PROSPECT ATC Prediction Model

This file defines the ATC prediction framework based on:
1. Graph Attention Networks (GAT)
2. Graph Convolutional Networks (GCN)
3. Kolmogorov-Arnold Networks (KAN)
4. Transfer Learning

Author: Duzhenshun
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import global_max_pool as gmp

from KANLinear import KANLinear


class KAN_PROSPECT_ATC(nn.Module):
    def __init__(self, num_features_xd=78, 
                       n_output=1, 
                       num_features_xt=1287,
                       n_filters=64, 
                       embed_dim=128,
                       output_dim=128, 
                       dropout1=0.1,
                       dropout2=0.2,
                       grid_size=3,  
                       spline_order=2,
                       scale_noise=0.2,
                       scale_base=0.2,
                       scale_spline=0.2,
                       weight_decay=1e-4): 
        super(KAN_PROSPECT_ATC, self).__init__()

        self.gcn1 = GATConv(num_features_xd, num_features_xd,heads=8,dropout=dropout1)
        self.bn1 = nn.BatchNorm1d(num_features_xd* 8)  
        self.gcn2= GCNConv(num_features_xd* 8, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim) 
        self.gcn3 = GCNConv(output_dim, 1024)
        self.bn3 = nn.BatchNorm1d(1024)  
        self.gcn4 = GCNConv(1024, output_dim)
        self.bn4 = nn.BatchNorm1d(output_dim) 
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        
        self.conv_xt1 = nn.Conv1d(in_channels=32, out_channels=n_filters, kernel_size=5, padding=2)
        self.bn_xt1 = nn.BatchNorm1d(n_filters)  
        self.fc_xt1 = KANLinear(n_filters * 128, output_dim)
        
        self.fc1 = KANLinear(2 * output_dim, output_dim, grid_size=grid_size, spline_order=spline_order,
                             scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline)
        
        self.bn_fc1 = nn.BatchNorm1d(output_dim)  
        
        self.fc3 = KANLinear(output_dim, 512, grid_size=grid_size, spline_order=spline_order,
                             scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline)
        self.bn_fc3 = nn.BatchNorm1d(512)  
        
        self.fc2 = KANLinear(512, 256, grid_size=grid_size, spline_order=spline_order,
                             scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline)
        self.bn_fc2 = nn.BatchNorm1d(256)  
        self.out = KANLinear(256, n_output, grid_size=grid_size, spline_order=spline_order,
                             scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline)

       
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout2)
        self.weight_decay = weight_decay  

    def forward(self, data):
       
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = self.bn1(x)  
        x = F.dropout(x, p=0.1)
        x = self.gcn2(x, edge_index)
        x = self.bn2(x) 
        x = self.gcn3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.gcn4(x, edge_index)
        
        x = self.bn4(x)
        
        x = self.relu(x)
        x = gmp(x, batch)  
        x1 = self.fc_g1(x)
        x = self.relu(x1)
        
        x = self.dropout(x)

       
        target = data.target
        
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt)
        conv_xt = self.bn_xt1(conv_xt)  
        conv_xt = self.relu(conv_xt)
        
        xt = conv_xt.view(-1, 64 * 128)  
        xt = self.fc_xt1(xt)
        xt = self.relu(xt)
        
        xc = torch.cat((x, xt), 1)

       
        xc = self.fc1(xc)
        xc = self.bn_fc1(xc)  
        xc = self.dropout(xc)
        
        xc = self.fc3(xc)
        xc = self.bn_fc3(xc) 
        
        xc = self.fc2(xc)
        xc = self.bn_fc2(xc)  
        out = self.out(xc)
        
        return out

"""
KAN_PROSPECT ADR Prediction Model

This file defines the ADR prediction framework based on:
1. Graph Attention Networks (GAT)
2. Graph Convolutional Networks (GCN)
3. Kolmogorov-Arnold Networks (KAN)
4. Multi-branch Conv1D feature extraction
5. Transfer Learning

Author: Duzhenshun
"""

class KAN_PROSPECT_ADR(nn.Module):
    def __init__(self, num_features_xd=78, 
                        n_output=1, 
                        num_features_xt=114,
                        n_filters=32, 
                        embed_dim=128,
                        output_dim=128, 
                        dropout1=0.3,
                        dropout2=0.3,
                        grid_size=10,  
                        spline_order=2,
                        scale_noise=0.05,
                        scale_base=1,
                        scale_spline=0.5,
                        weight_decay=1e-4):  
         super(KAN_PROSPECT_ADR, self).__init__()
         
         self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout1)
         self.bn1 = nn.BatchNorm1d(num_features_xd * 10)  
         self.gcn2 = GATConv(num_features_xd * 10, 256, dropout=dropout1)
         self.bn2 = nn.BatchNorm1d(256)  
         self.gcn3 = GCNConv(256, 1024)
         self.bn3 = nn.BatchNorm1d(1024)
         self.gcn4 = GATConv(1024, output_dim, dropout=dropout1)
         
         self.bn4 = nn.BatchNorm1d(output_dim) 
         self.fc_g1 = KANLinear(output_dim, output_dim, grid_size=grid_size, spline_order=spline_order, 
                                scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline)
         
         self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
         self.conv_xt1 = nn.Conv1d(in_channels=16, out_channels=n_filters, kernel_size=8)
         self.bn_xt1 = nn.BatchNorm1d(n_filters)  
         self.fc_xt1 = KANLinear(n_filters * 121, output_dim, grid_size=grid_size, spline_order=spline_order,
                                 scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline)
         self.conv_xt2 = nn.Conv1d(in_channels=16, out_channels=n_filters, kernel_size=8)
         self.bn_xt2 = nn.BatchNorm1d(n_filters)
         self.fc_xt2 = KANLinear(n_filters * 121, 256, grid_size=grid_size, spline_order=spline_order,
                                 scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline)
         self.fc_xt3 = KANLinear(output_dim+256, 256, grid_size=grid_size, spline_order=spline_order,
                                 scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline)
         self.fc_xt4 = KANLinear(256, output_dim, grid_size=grid_size, spline_order=spline_order,
                                 scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline)
         
         self.fc1 = KANLinear(2 * output_dim, 1024, grid_size=grid_size, spline_order=spline_order,
                              scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline)
         self.bn_fc1 = nn.BatchNorm1d(1024)  
         self.fc2 = KANLinear(1024, 256, grid_size=grid_size, spline_order=spline_order,
                              scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline)
         self.bn_fc2 = nn.BatchNorm1d(256) 
         self.fc3 = KANLinear(256, 256, grid_size=grid_size, spline_order=spline_order,
                              scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline)
         
         self.out = KANLinear(256, n_output, grid_size=grid_size, spline_order=spline_order,
                              scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline)
         
         self.relu = nn.ReLU()
         self.dropout = nn.Dropout(dropout2)
         self.weight_decay = weight_decay  
    def forward(self, data):
         
         x, edge_index, batch = data.x, data.edge_index, data.batch
         x = F.dropout(x, p=0.2, training=self.training)
         x = F.elu(self.gcn1(x, edge_index))
         x = self.bn1(x)  
         x = F.dropout(x, p=0.2, training=self.training)
         x = self.gcn2(x, edge_index)
         x = self.bn2(x)  
         x = F.dropout(x, p=0.2, training=self.training)
         x = self.gcn3(x, edge_index)
         x = self.bn3(x)
         x = self.relu(x)
         x = self.gcn4(x, edge_index)
        
         x = self.relu(x)
         x = gmp(x, batch) 
         x1 = self.fc_g1(x)
         x = self.relu(x1)
         
         
         target = data.target
         embedded_xt = self.embedding_xt(target)
         
         conv_xt = self.conv_xt1(embedded_xt)
         conv_xt = self.bn_xt1(conv_xt)  
         conv_xt = self.relu(conv_xt)
         
         xt = conv_xt.view(-1, 32 * 121)  
         xt = self.fc_xt1(xt)
         
         xt = self.relu(xt)
         conv_xt1 = self.conv_xt2(embedded_xt)
         conv_xt1 = self.bn_xt2(conv_xt1) 
         conv_xt1 = self.relu(conv_xt1)
         xt1 = conv_xt1.view(-1, 32 * 121)
         xt1 = self.fc_xt2(xt1)
         xt = F.dropout(xt, p=0.1, training=self.training)
        
         xt1 = self.relu(xt1)
         xt = torch.cat((xt, xt1), 1)
         
         xt = self.relu(xt)
         xt = self.fc_xt3(xt)
         xt = F.dropout(xt, p=0.2, training=self.training)
         xt = self.relu(xt)
         xt = self.fc_xt4(xt)
         
         xc = torch.cat((x, xt), 1)
         
         xc = self.fc1(xc)
         xc = self.bn_fc1(xc)  
         xc = self.relu(xc)
         xc = self.dropout(xc)
         xc = self.fc2(xc)
         xc = self.bn_fc2(xc) 
         xc = self.relu(xc)
         xc = self.fc3(xc)
         
         out = self.out(xc)
         
         return out