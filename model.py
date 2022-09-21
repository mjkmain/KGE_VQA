import cv2
import torch
import torch.nn as nn
import transformers
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import json
import torchvision.models as models

'''
for vision transformer
'''
import timm
from transformers import logging
logging.set_verbosity_error()
logging.set_verbosity_warning()

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VQAModel(nn.Module):
    def __init__(self, num_target, dim_i, dim_h=1024, config=None):
        super(VQAModel, self).__init__()
        
        print("-----  ","%20s"%"Load pretrained BERT Model", "   -----")
        
        self.bert = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base')
        
        if config.image_model == 'vit':
            print("-----  ","%20s"%"Load pretrained VIT Model", "    -----\n")
            
            self.i_model = timm.create_model('vit_base_patch16_224', pretrained=True) 
            self.i_model.head = nn.Linear(768, dim_i) 
            
        elif config.image_model == 'resnet':
            print("-----  ","%20s"%"Load pretrained ResNet Model", " -----\n")
            
            self.i_model = models.resnet50(pretrained=True)
            self.i_model.fc = nn.Linear(self.i_model.fc.in_features, dim_i)
        
        self.i_relu = nn.ReLU()
        self.drop = nn.Dropout(config.drop_out)
        
        #classfier: MLP기반의 분류기를 생성
        self.linear1 = nn.Linear(dim_i, dim_h)
        
        self.q_relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_h, num_target)
        
        
    def forward(self, idx, mask, image, config, *emb):
        
        q_f = self.bert(idx, mask) #질문을 Bert를 활용해 Vector화
        q_f = q_f.pooler_output
        i_f = self.drop(self.i_model(image)) # 이미지를 Vision Transformer를 활용해 Vector화
        
        iq_f = i_f*q_f 
        
        if config.use_kge:
            emb_1 = emb[0].squeeze()
            emb_2 = emb[1].squeeze()
            embed = emb_1*emb_2


            uni_f = embed*iq_f
            
        else:
            uni_f = iq_f
            
        return self.linear2(self.drop(self.q_relu(self.drop(self.linear1(uni_f))))) #MLP classfier로 답변 예측