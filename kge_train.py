from torch.optim import Adam

from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.models import TransEModel
from torchkge.utils.datasets import load_fb15k
from torchkge.utils import Trainer, MarginLoss

import os
import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.optim import Adam
from tqdm import tqdm

from torchkge.models import TransEModel , TransRModel, ComplExModel, DistMultModel, ConvKBModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, BinaryCrossEntropyLoss , DataLoader
from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.data_structures import KnowledgeGraph
from torchkge.inference import RelationInference

import pickle
import gc
gc.collect()
cuda.empty_cache()

def train_kge(epochs=10000, lr=1e-4, batch_size=64, margin=0.5, conv_size=3):

    config = {
        'emb_dim' : 256,
        'ent_emb_dim' : 256,
        'rel_emb_dim' : 256,
        'lr' : lr,
        'epochs' : epochs,
        'batch_size' : batch_size,
        'margin' : margin,
    }

    df = pd.read_csv('./data/triple', index_col=0)

    df2 = df.copy()
    df2.columns = ['from','rel','to']
    df2.head(1)

    usermap = {user : i for i , user in enumerate(df['head'])}
    ratingmap = {rate : i for i , rate in enumerate(df['relation'])}
    itemmap = {item : i for i , item in enumerate(df['tail'])}

    entity_ori = np.concatenate([df['head'].values ,df['tail'].values])
    entitymap = {entity : i for i , entity in enumerate(entity_ori)}

    df2['rel'].map(lambda x : ratingmap.get(x))
    df2['from'].map(lambda x : entitymap.get(x))
    df2['to'].map(lambda x : entitymap.get(x))

    print(f'originally counts {len(df)}\n\nuser nunique {len(usermap)}, item nunique {len(itemmap)}')

    kg = KnowledgeGraph(df2)
    kg_train, kg_val, kg_test = kg.split_kg(share = 0.8,
                                            validation=True)

    # Define the model and criterion
    model_convKB = ConvKBModel(config.get('emb_dim'),
                            conv_size,
                            kg_train.n_ent,
                            kg_train.n_rel,
                            )
    criterion = MarginLoss(config.get('margin'))

    # Move everything to CUDA if available
    if cuda.is_available():
        cuda.empty_cache()
        model_convKB.cuda()
        criterion.cuda()

    # Define the torch optimizer to be used
    optimizer = Adam(model_convKB.parameters(), lr=config.get('lr'), weight_decay=1e-5)

    sampler = BernoulliNegativeSampler(kg_train)
    dataloader = DataLoader(kg_train, batch_size=config.get('batch_size'))
    iterator = tqdm(range(epochs), unit='epoch')

    for epoch in iterator:
        running_loss = 0.0
        
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            n_h, n_t = sampler.corrupt_batch(h, t, r)

            optimizer.zero_grad()

            # forward + backward + optimize
            pos, neg = model_convKB(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        iterator.set_description(
            'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                                running_loss / len(dataloader)))

    model_convKB.normalize_parameters()
    
    kge_config =   {'emb_dim' : config.get('emb_dim'), 
                    'conv_size' : conv_size,
                    'n_ent' : kg_train.n_ent,
                    'n_rel' : kg_train.n_rel,
                    'epoch' : epochs,
                    'lr' : lr,
                    'batch_size' : batch_size,
                    'margin' : margin,
                    'conv_size' : conv_size}
    
    with open(f'./kge_save/{epochs}_{lr}_{batch_size}_{margin}_{conv_size}_kg.pkl', 'wb') as f:
        pickle.dump(kg, f)
        
    with open(f'./kge_save/{epochs}_{lr}_{batch_size}_{margin}_{conv_size}_kge_config.pkl', 'wb') as f:
        pickle.dump(kge_config, f)
        

    torch.save(model_convKB.state_dict(), f'./kge_save/{epochs}_{lr}_{batch_size}_{margin}_{conv_size}_convKB.pt')
    
    
    return kg, model_convKB