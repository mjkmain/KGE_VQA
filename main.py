from pickletools import optimize
from urllib.robotparser import RequestRate
from dataloader import VQADataset
from model import VQAModel
from kge_train import train_kge
from vqa_train import train
from utils import get_num_target, get_test_score, str2bool

import torch
import argparse
from transformers import AutoTokenizer
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from torchkge.models import ConvKBModel
import pickle 
import os

from transformers import logging

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type = int, required=False, default=10)
    parser.add_argument('--drop_out', type = float, required=False, default=0.2)
    parser.add_argument('--batch_size', type = int, required=False, default=128)
    parser.add_argument('--lr', type = float, required=False, default=5e-4)
    parser.add_argument('--use_kge', type=str2bool, required=False, default=True,  help='True : Train VQA Model with KG Embedding')
    parser.add_argument('--use_kge_pt', type=str2bool, required=False, default=True,  help='True : Use pretrained KGE Model')
    parser.add_argument('--image_model', type=str, required=False, default='vit', choices=['vit', 'resnet'])
    parser.add_argument('--device', required=False, default='cuda:0')

    parser.add_argument('--ans1_only', required=False, default=True)
    parser.add_argument('--ans2_only', required=False, default=False)
    parser.add_argument('--both', required=False, default=False)
    parser.add_argument('--max_token', required=False, default=30)
    
    '''
    KGE
    '''
    parser.add_argument('--kge_n_epoch', type=int, required=False, default=10000)
    parser.add_argument('--kge_lr', type=float, required=False, default=1e-4)
    parser.add_argument('--kge_batch', type=int, required=False, default=64)
    parser.add_argument('--kge_margin', type=float, required=False, default=0.5)
    parser.add_argument('--kge_conv_size', required=False, default=3)
    config = parser.parse_args()
    return config

if __name__ == '__main__':
    config = parse_args()
    logging.set_verbosity_error()
    
    if config.use_kge:
        if not config.use_kge_pt:
            print("\n-----     ","%20s"%"Training KGE Model", "       -----")
            
            kg, model_convKB = train_kge(epochs=config.kge_n_epoch, 
                                            lr=config.kge_lr, 
                                            batch_size=config.kge_batch,
                                            margin=config.kge_margin,
                                            conv_size=config.kge_conv_size
                                            )
        
        if config.use_kge_pt:
            print("\n-----  ","%20s"%"Load pretrained KGE Model", "    -----")
            with open(f'./kge_save/{config.kge_n_epoch}_{config.kge_lr}_{config.kge_batch}_{config.kge_margin}_{config.kge_conv_size}_kge_config.pkl', 'rb') as f:
                kge_config = pickle.load(f)
                
                
            with open(f'./kge_save/{config.kge_n_epoch}_{config.kge_lr}_{config.kge_batch}_{config.kge_margin}_{config.kge_conv_size}_kg.pkl', 'rb') as f:
                kg = pickle.load(f)
                
            model_convKB = ConvKBModel(kge_config['emb_dim'],
                                        kge_config['conv_size'],
                                        kge_config['n_ent'],
                                        kge_config['n_rel']
                                        )
            model_convKB.load_state_dict(torch.load(f'./kge_save/{config.kge_n_epoch}_{config.kge_lr}_{config.kge_batch}_{config.kge_margin}_{config.kge_conv_size}_convKB.pt'))
            
            
            
        emb_entity_ = model_convKB.get_embeddings()[0].detach().cpu().numpy()
        emb_rel_ = model_convKB.get_embeddings()[1].detach().cpu().numpy()
        '''
        make 2 [UNK] token for answer2 randomly initialization
        '''
        
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    model = VQAModel(get_num_target(config), dim_i=768, dim_h=1024, config=config)
    model = torch.nn.DataParallel(model)
    model = model.to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    if config.ans1_only:
        data_df = pd.read_csv('./data/data_v3_triple.csv', index_col=0)

        
        train_df, test_df = train_test_split(data_df, test_size=0.2)
        ans_list = train_df['answer'].value_counts().reset_index()
        ans_list.columns=['answer', 'count']    
        
        
        train_df, valid_df = train_test_split(train_df, test_size=0.25)
        
        
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        test_df.to_csv('./data/test_df.csv')
        
    if config.use_kge:
        train_dataset = VQADataset(tokenizer, train_df, ans_list, config.max_token, transform, config, emb_entity_, emb_rel_, kg)
        train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=22, shuffle=True, pin_memory=True)
        valid_dataset = VQADataset(tokenizer, valid_df, ans_list, config.max_token, transform, config, emb_entity_, emb_rel_, kg) 
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size , num_workers=22, shuffle=False, pin_memory=True)
    if not config.use_kge:
        train_dataset = VQADataset(tokenizer, train_df, ans_list, config.max_token, transform, config)
        train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=22, shuffle=True, pin_memory=True)
        valid_dataset = VQADataset(tokenizer, valid_df, ans_list, config.max_token, transform, config) 
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size , num_workers=22, shuffle=False, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()

    
    print("-----   ","%20s"%"Start Training ", "        -----")

    print(f"\n# Train data : {len(train_df)}")
    print(f"# Valid data : {len(valid_df)}")
    print(f"# Test data : {len(test_df)}")
    valid_str = train(model, train_loader, valid_loader, criterion, optimizer, config.device, config.n_epoch, config)
    

    print("\nVQA MODEL CONFIG")
    print(f"image_model : {config.image_model}, drop_out : {config.drop_out}, batch_size : {config.batch_size}, learning_rate : {config.lr}, use_kge : {config.use_kge}")
    print("\nKGE MODEL CONFIG")
    if config.use_kge:
        print(f"kge_n_epoch : {config.kge_n_epoch}, kge_lr : {config.kge_lr}, kge_batch : {config.kge_batch}, kge_margin : {config.kge_margin}, kge_conv_size : {config.kge_conv_size}\n")
    if not config.use_kge:
        print("use_kge : False")
    print("\n-----   ","%20s"%"Start Testing ", "        -----")
    
    if config.use_kge:
        result_str = get_test_score(model, test_df, tokenizer, transform, ans_list, config.device, config, emb_entity_, emb_rel_, kg)

    if not config.use_kge:
        result_str = get_test_score(model, test_df, tokenizer, transform, ans_list, config.device, config)
        
    result_str += valid_str
    result_str += f"image_model : {config.image_model}, drop_out : {config.drop_out}, batch_size : {config.batch_size}, learning_rate : {config.lr}, use_kge : {config.use_kge}, kge_n_epoch : {config.kge_n_epoch}, kge_lr : {config.kge_lr}, kge_batch : {config.kge_batch}, kge_margin : {config.kge_margin}, kge_conv_size : {config.kge_conv_size}\n" 
    
    with open("./results/result.txt", "a") as f:
        f.write(result_str) 