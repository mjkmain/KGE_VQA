from tqdm import tqdm 
import torch 
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt



def train(model, train_loader, valid_loader, criterion, optimizer, device, n_epoch, config):
    total_train_loss = []
    total_train_acc = []
    total_valid_loss = []
    total_valid_acc = []
    
    best_epoch = 0
    best_acc = 0
    
    for epoch in range(1, n_epoch+1):
        
        train_count_correct = 0
        valid_count_correct = 0
        
        train_total_num = 0
        valid_total_num = 0
        
        train_loss = 0
        valid_loss = 0
        
        model.train()
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), leave=False): #학습 데이터를 batch size만큼씩 읽어옴
            optimizer.zero_grad()
            imgs = batch['image'].to(device)  #이미지
            q_bert_ids = batch['q_ids'].to(device) #질문
            q_bert_mask = batch['q_mask'].to(device) 
            
            if config.use_kge:
                emb_1 = batch['emb_1'].to(device)
                emb_2 = batch['emb_2'].to(device)
                outputs = model(q_bert_ids, q_bert_mask, imgs, config, emb_1, emb_2) 
            else:
                outputs = model(q_bert_ids, q_bert_mask, imgs, config)
                        
            answers = batch['answer'].to(device) #응답
            answers = answers.squeeze()
            
            loss = criterion(outputs, answers) #예측된 답변과 실제 정답과 비교하여 loss계산

            train_loss += float(loss)
            loss.backward(loss)
            optimizer.step()
            
            predicted = torch.argmax(outputs, dim=1)
            count_correct = np.count_nonzero((np.array(predicted.cpu())==np.array(answers.cpu())) == True) #정답갯수를 계산
            train_count_correct += count_correct
            train_total_num += answers.size(0)
            
        train_loss /= len(train_loader)
        train_acc = train_count_correct/train_total_num
        
        model.eval()
        for idx, batch in tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False):
            imgs = batch['image'].to(device)
            q_bert_ids = batch['q_ids'].to(device)
            q_bert_mask = batch['q_mask'].to(device)
            
            if config.use_kge:
                emb_1 = batch['emb_1'].to(device)
                emb_2 = batch['emb_2'].to(device)
                outputs = model(q_bert_ids, q_bert_mask, imgs, config, emb_1, emb_2)
            
            else:
                outputs = model(q_bert_ids, q_bert_mask, imgs, config)
                
            answers = batch['answer'].to(device)
            answers = answers.squeeze()
            
            
            loss = criterion(outputs, answers)
            valid_loss += float(loss)
            
            predicted = torch.argmax(outputs, dim=1)
            count_correct = np.count_nonzero((np.array(predicted.cpu())==np.array(answers.cpu())) == True) #정답갯수를 계산
            valid_count_correct += count_correct
            valid_total_num += answers.size(0)
            
        valid_loss /= len(valid_loader)
        valid_acc = valid_count_correct/valid_total_num
            
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            best_acc_model = deepcopy(model.state_dict())
            
        total_train_loss.append(train_loss)
        total_valid_loss.append(valid_loss)
        total_train_acc.append(train_acc*100)
        total_valid_acc.append(valid_acc*100)
            
        print(f"[{epoch:2}/{n_epoch:2}] TRAIN LOSS: {train_loss:.4f} TRAIN ACC: {train_acc:.4f} | VALID LOSS: {valid_loss:.4f} VALID ACC: {valid_acc:.4f} | BEST VALID ACC: {best_acc:.4f} |")
    print(f"\nSave the best acc model in epoch {best_epoch}")
    torch.save(best_acc_model, f'./models/{config.image_model}_epoch{best_epoch}_acc{best_acc*100:.2f}.pt')

    epochs = np.arange(0,n_epoch)

    fig = plt.figure(figsize=(24, 10))
    fig.add_subplot(1, 2, 1)
    plt.title(f'{config.image_model}\nLoss', fontsize=20) 
    plt.ylim(0, 8)
    plt.plot(epochs, total_train_loss, label="train") 
    plt.plot(epochs, total_valid_loss, label="valid") 
    plt.legend()
    
    fig.add_subplot(1, 2, 2)
    plt.title(f'{config.image_model}\nAcc', fontsize=20) 
    plt.plot(epochs, total_train_acc, label="train") 
    plt.plot(epochs, total_valid_acc, label="valid")     
    plt.ylim(0, 100)
    plt.legend()
    
    fig.savefig(f"./results/{config.image_model}_acc{best_acc*100:.2f}_epoch{best_epoch}.png")
    
    return f'Valid Acc : {best_acc*100:.2f}, epoch : {best_epoch}, '