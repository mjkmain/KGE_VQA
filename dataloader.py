import torch
from PIL import Image

from utils import get_embedded_vec

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, answer_list, max_token, transform, config, *kge_args):
        
        self.tokenizer = tokenizer
        self.data = data
        self.max_token = max_token
        self.answer_list = answer_list        
        self.transform = transform
        self.config = config
        
        if self.config.use_kge:        
            self.emb_entity_ = kge_args[0]
            self.emb_rel_ = kge_args[1]
            self.kg = kge_args[2]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        question = self.data['question'][index] #질문
        answer = self.data['answer'][index]  #응답
        img_loc = self.data['image_path'][index] #사진파일
        
        if self.config.use_kge:
            embbed_1, embbed_2 = get_embedded_vec(self.data, index, self.emb_entity_, self.emb_rel_, self.kg)
        
        tokenized = self.tokenizer.encode_plus("".join(question),
                                     None,
                                     add_special_tokens=True,
                                     max_length = self.max_token,
                                     truncation=True,
                                     pad_to_max_length = True
                                              )
        
        
        ids = tokenized['input_ids']
        mask = tokenized['attention_mask']
        image = Image.open(img_loc).convert('RGB')  #이미지 데이터를 RGB형태로 읽음 질문을 tokenize한다.
        image = self.transform(image)  #이미지 데이터의 크기 및 각도등을 변경
        
        answer_ids = self.answer_list[self.answer_list['answer']==answer].index #응답을 숫자 index로 변경, e.g.) "예"-->0 "아니요" --> 1
        if len(answer_ids)==0:
            answer_ids = self.answer_list[self.answer_list['answer']=="예"].index

        if self.config.use_kge:
            return {'q_ids': torch.tensor(ids, dtype=torch.long), 
                    'q_mask': torch.tensor(mask, dtype=torch.long),
                    'answer': torch.tensor(answer_ids, dtype=torch.long),
                    'image': image,
                    'emb_1' : embbed_1,
                    'emb_2' : embbed_2}
        else:
            return{'q_ids': torch.tensor(ids, dtype=torch.long), 
                    'q_mask': torch.tensor(mask, dtype=torch.long),
                    'answer': torch.tensor(answer_ids, dtype=torch.long),
                    'image': image}