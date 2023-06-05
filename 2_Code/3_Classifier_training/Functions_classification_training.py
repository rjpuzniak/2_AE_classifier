#!/usr/bin/python3

import numpy as np

from collections import OrderedDict

import torch
import torch.nn as nn

# 1-layer U-Net architecture

class UNet_1_layer(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1, init_features=64, scaling=2):
        super(UNet_1_layer, self).__init__()
                
        # Encoding layers
        self.encoder1 = self.unet_block(in_channels, init_features, "enc1")
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)

        # Bottleneck layer
        self.bottleneck = self.unet_block(init_features, init_features*scaling, name='bottleneck')
        
        # Decoding layers (where merge with prevois encoding layers occurs)          
        self.upconv1 = nn.ConvTranspose3d(init_features*scaling, init_features, kernel_size=2, stride=2)
        self.decoder1 = self.unet_block(init_features, init_features, name='dec1')
        
        # Final convolution - output equals number of output channels
        self.conv = nn.Conv3d(init_features, out_channels, kernel_size=1) 
        
    def forward(self,x):
        
        # Encoding
        enc1 = self.encoder1(x)
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool1(enc1))
        
        # Upconvolving, concatenating data from respective encoding phase and executing UNet block
        dec1 = self.upconv1(bottleneck)
        dec1 = self.decoder1(dec1)
        
        out_conv = self.conv(dec1)
        
        return torch.sigmoid(out_conv)
    
    def unet_block(self, in_channels, features, name):
        
        return nn.Sequential(OrderedDict([(name+'conv1',nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False)),
                             (name+'bnorm1', nn.BatchNorm3d(num_features=features)),
                             (name+'relu1', nn.ReLU(inplace=True)),
                             (name+'conv2', nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)),
                             (name+'bnorm2', nn.BatchNorm3d(num_features=features)),
                             (name+'relu2', nn.ReLU(inplace=True))])
                            )

    def output_latent_representations(self,x):
        
        # Encoding
        enc1 = self.encoder1(x)
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool1(enc1))
                
        return bottleneck

# 2-layer U-Net architecture

class UNet_2_layer(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1, init_features=10, scaling=2):
        super(UNet_2_layer, self).__init__()
                
        # Encoding layers
        self.encoder1 = self.unet_block(in_channels, init_features, "enc1")
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        self.encoder2 = self.unet_block(init_features, init_features*scaling, name='enc2')
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)

        # Bottleneck layer
        self.bottleneck = self.unet_block(init_features*scaling, init_features*scaling**2, name='bottleneck')
        
        # Decoding layers (where merge with prevois encoding layers occurs)        
        self.upconv2 = nn.ConvTranspose3d(init_features*scaling**2, init_features*scaling, kernel_size=2, stride=2)
        self.decoder2 = self.unet_block(init_features*scaling, init_features*scaling, name='dec2')
                
        self.upconv1 = nn.ConvTranspose3d(init_features*scaling, init_features, kernel_size=2, stride=2)
        self.decoder1 = self.unet_block(init_features, init_features, name='dec1')
        
        # Final convolution - output equals number of output channels
        self.conv = nn.Conv3d(init_features, out_channels, kernel_size=1) 
        
    def forward(self,x):
        
        # Encoding
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))

        # Upconvolving, concatenating data from respective encoding phase and executing UNet block
        dec2 = self.upconv2(bottleneck)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(dec1)
        
        out_conv = self.conv(dec1)
        
        return torch.sigmoid(out_conv)
    
    def unet_block(self, in_channels, features, name):
        
        return nn.Sequential(OrderedDict([(name+'conv1',nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False)),
                             (name+'bnorm1', nn.BatchNorm3d(num_features=features)),
                             (name+'relu1', nn.ReLU(inplace=True)),
                             (name+'conv2', nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)),
                             (name+'bnorm2', nn.BatchNorm3d(num_features=features)),
                             (name+'relu2', nn.ReLU(inplace=True))])
                            )

    def output_latent_representations(self,x):

        # Encoding
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))
                
        return bottleneck

# Network used for classification
class Classifier(nn.Module):
    
    def __init__(self, extraction_layers, init_features, scaling, num_fc_layers, num_hidden_nodes, weights_path='../../1_Data/2_Trained_AE/'):
        super(Classifier, self).__init__()
        
        self.extraction_layers = extraction_layers
        self.init_features = init_features
        self.scaling = scaling
        self.num_fc_layers = num_fc_layers
        self.num_hidden_nodes = num_hidden_nodes
        
        # Initialize the network
        if self.extraction_layers == 1:
            network_extracting_features = UNet_1_layer(1,1,self.init_features, self.scaling)
            network_extracting_features.load_state_dict(torch.load(weights_path+'/1_layer_'+str(self.init_features)+'_'+str(self.scaling)+'/optimal_weights'))
        elif self.extraction_layers == 2:
            network_extracting_features = UNet_2_layer(1,1,self.init_features, self.scaling)
            network_extracting_features.load_state_dict(torch.load(weights_path+'/2_layer_'+str(self.init_features)+'_'+str(self.scaling)+'/optimal_weights'))
            
        child = network_extracting_features.children()
        
        # Copy the analysis stream from network_extracing_featuresabs
        if self.extraction_layers ==1:
            self.feature_extraction = nn.Sequential(*list(child)[:3])
        elif self.extraction_layers ==2:
            self.feature_extraction = nn.Sequential(*list(child)[:5])
            
        # Create classification layers
        if self.extraction_layers == 1:
            if self.num_fc_layers==0:
                self.classifier = nn.Linear(self.init_features*self.scaling*12*12*4,1)
            elif self.num_fc_layers==1:
                self.classifier = nn.Sequential(nn.Linear(self.init_features*self.scaling*12*12*4,self.num_hidden_nodes),nn.ReLU(),nn.Linear(self.num_hidden_nodes,1))
        elif self.extraction_layers == 2:
            if self.num_fc_layers==0:
                self.classifier = nn.Linear(self.init_features*(self.scaling**2)*6*6*2,1)
            elif self.num_fc_layers==1:
                self.classifier = nn.Sequential(nn.Linear(self.init_features*(self.scaling**2)*6*6*2,self.num_hidden_nodes),nn.ReLU(),nn.Linear(self.num_hidden_nodes,1))
        
     # Freeze all the layers apart from the classifying one
    def freeze_feature_extraction(self):

        counter = 0
        for layer in self.children():
            counter += 1
            if counter < 2:
                #print(layer)
                for param in layer.parameters():
                    param.requires_grad = False

    # Freeze feature selection layers only
    def freeze_classification(self):

        counter = 0
        for layer in self.children():
            counter += 1
            if counter == 2:
                #print(layer)
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self,x):

            # Feature extraction
            x = self.feature_extraction(x)

            # Flatten the image
            x = x.view((x.shape[0], -1))        

            # Classifying FC layer and activation function
            x = self.classifier(x)

            return torch.sigmoid(x)

    # Output latent representations (or activations) of the last CNN layer
    def output_latent_representations(self,x):        
        return self.feature_extraction(x)     
