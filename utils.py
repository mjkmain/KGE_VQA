import torch 
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse

def get_embedded_vec(data, index, emb_entity_, emb_rel_, kg):
    if len(data['fact'][index].split(':')) == 1:
        head_1 = data['fact'][index].split(':')[0].split("'")[1]
        relation_1 = data['fact'][index].split(':')[0].split("'")[3]
        tail_1 = data['fact'][index].split(':')[0].split("'")[5]
        
        head_2 = data['fact'][index].split(':')[0].split("'")[1]
        relation_2 = data['fact'][index].split(':')[0].split("'")[3]
        tail_2 = data['fact'][index].split(':')[0].split("'")[5]
                    
    elif len(data['fact'][index].split(':')) == 2:
        head_1 = data['fact'][index].split(':')[0].split("'")[1]
        relation_1 = data['fact'][index].split(':')[0].split("'")[3]
        tail_1 = data['fact'][index].split(':')[0].split("'")[5]
        
        head_2 = data['fact'][index].split(':')[1].split("'")[1]
        relation_2 = data['fact'][index].split(':')[1].split("'")[3]
        tail_2 = data['fact'][index].split(':')[1].split("'")[5]
    
    head_1_emb = emb_entity_[kg.ent2ix[head_1]]
    rel_1_emb = emb_rel_[kg.rel2ix[relation_1]]
    tail_1_emb = emb_entity_[kg.ent2ix[tail_1]]
    
    head_2_emb = emb_entity_[kg.ent2ix[head_2]]
    rel_2_emb = emb_rel_[kg.rel2ix[relation_2]]
    tail_2_emb = emb_entity_[kg.ent2ix[tail_2]]
        
    return torch.tensor([head_1_emb, rel_1_emb, tail_1_emb], dtype=torch.float32).reshape(-1,1),\
        torch.tensor([head_2_emb, rel_2_emb, tail_2_emb], dtype=torch.float32).reshape(-1,1)
        
        
def get_num_target(config=None):
    if config.ans1_only:
        data_df = pd.read_csv('./data/data_v3_triple.csv', index_col=0)
        ans_list = data_df['answer'].value_counts().reset_index()
        ans_list.columns=['answer', 'count']
        return len(ans_list)
    
def get_test_score(model, test_df, tokenizer, transform, ans_list, device, config, *kge_args):
    correct_count = 0
    total_count = 0
    
    for index in tqdm(range(len(test_df))):
        img_file = test_df['image_path'][index]
        question = test_df['question'][index]
        if config.use_kge:
            embbed_1, embbed_2 = get_embedded_vec(test_df, index, kge_args[0], kge_args[1], kge_args[2])
    
        model.eval()
        img = transform(Image.open(img_file).convert("RGB")).unsqueeze(0)
        img = img.to(device)
        encoded = tokenizer.encode_plus("".join(question),
                                        None,
                                        add_special_tokens=True,
                                        max_length=30,
                                        truncation=True,
                                        pad_to_max_length=True)

        q_bert_ids, q_bert_mask = encoded['input_ids'], encoded['attention_mask']
        q_bert_ids = torch.tensor(q_bert_ids, dtype=torch.long).unsqueeze(0).to(device)
        q_bert_mask = torch.tensor(q_bert_mask, dtype=torch.long).unsqueeze(0).to(device)
        if config.use_kge:
            output = model(q_bert_ids, q_bert_mask, img, config, embbed_1, embbed_2)
        if not config.use_kge:
            output = model(q_bert_ids, q_bert_mask, img, config)
            
        predicted = torch.argmax(output, dim=1).item()
        
        ans = ans_list['answer'].iloc[predicted]
        if test_df['answer'][index] == ans:
            correct_count += 1
        total_count += 1
        
    print(f"Test Score : {correct_count/total_count * 100:2f}")
    return f'Test acc : {correct_count/total_count * 100:2f}'


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')