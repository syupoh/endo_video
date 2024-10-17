import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset, ConcatDataset
import torch.nn.functional as F

import os
import json
from PIL import Image
from torchvision import transforms

import argparse

import pdb
import time

import sys
from tqdm import tqdm
from datetime import datetime
from collections import Counter

from sklearn.metrics import classification_report, roc_auc_score

from wce_utils import DualOutput
# from wce_utils import DualOutput, save_best_checkpoint, argparse_set
from wce_model import PSDeVCEM, STPN2, PSDeVCEM2
from wce_dataloader import WCEVideoDataset


# class VideoTransformer(nn.Module):
#     def __init__(self, input_channels=3, embed_dim=512, num_heads=8, num_layers=4, hidden_dim=1024):
#         super(VideoTransformer, self).__init__()
        
#         # 3D CNN for spatial feature extraction
#         self.conv3d = nn.Sequential(
#             nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
#             nn.BatchNorm3d(64),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
#             nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
#             nn.BatchNorm3d(128),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
#             nn.Conv3d(128, embed_dim, kernel_size=(3, 3, 3), stride=1, padding=1),
#             nn.BatchNorm3d(embed_dim),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool3d((None, 7, 7))  # Reduce spatial dimensions to 7x7
#         )
        
#         # Transformer Encoder for temporal feature extraction
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.1)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Fully connected layer for classification
#         self.fc = nn.Linear(embed_dim, 1)

#     def forward(self, x):
#         # x shape: (batch_size, seq_length, channels, height, width)
#         batch_size, seq_length, c, h, w = x.size()

#         # Apply 3D CNN to extract spatial features
#         x = x.permute(0, 2, 1, 3, 4)  # (batch_size, channels, seq_length, height, width)
#         x = self.conv3d(x)  # (batch_size, embed_dim, seq_length, 7, 7)
#         x = x.flatten(3)  # Flatten spatial dimensions (batch_size, embed_dim, seq_length, 49)
#         x = x.permute(2, 0, 1, 3).flatten(2)  # (seq_length, batch_size, embed_dim)

#         # Apply Transformer to capture temporal relationships
#         x = self.transformer(x)  # (seq_length, batch_size, embed_dim)

#         # Use the output corresponding to the last sequence element
#         out = x[-1]  # (batch_size, embed_dim)

#         # Classification layer
#         out = self.fc(out)  # (batch_size, 1)

#         return out

# # TC loss 는 Video transformer 의 self-attention 으로 대체 가능
# class TemporalChangeLoss(nn.Module): 
#     def __init__(self, margin=1.0):
#         super(TemporalChangeLoss, self).__init__()
#         self.margin = margin
    
#     def forward(self, outputs, labels):
#         """
#         outputs: 모델의 출력 (batch_size, seq_length, feature_dim)
#         labels: 레이블 (batch_size, seq_length)
#         """
#         batch_size, seq_length, _ = outputs.size()
#         loss = 0.0
        
#         for i in range(seq_length - 1):
#             # 두 연속된 프레임 간의 차이 계산
#             temporal_diff = outputs[:, i+1, :] - outputs[:, i, :]
#             # 레이블의 변화가 있는 경우, 해당 차이에 따라 손실을 증가시킴
#             label_diff = labels[:, i+1] - labels[:, i]
#             label_diff = label_diff.abs()  # 변화의 절대값

#             # MSE를 사용하여 temporal difference에 대한 손실 계산
#             loss += label_diff * torch.sum(torch.square(temporal_diff))
        
#         loss /= (batch_size * (seq_length - 1))
#         return loss

def initialize_weights(model):
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)

# 체크포인트 저장 함수 (최적의 모델만 저장하고, 이전 체크포인트 삭제)
def save_best_checkpoint(args, epoch, model, optimizer, 
                         val_loss, best_accuracy, accuracy, scheduler=None, checkpoint_dir="checkpoints", best_checkpoint=None):
    
    if accuracy > best_accuracy:
        # 새로운 최적 체크포인트 저장
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, 
                                       f"{args.model_name}_{args.curtime}_{epoch+1}_{accuracy:02.02f}_gpu{args.gpu}.pth")
        
        torch.save({
            'epoch': epoch + 1,  # 현재 에포크
            'model_state_dict': model.state_dict(),  # 모델 가중치
            'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 상태
            'val_loss': val_loss,  # validation 손실 값
            'accuracy': accuracy,  # validation accuracy
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        }, checkpoint_path)

        print(f" New best model saved at {checkpoint_path}")

        # 이전 최적의 체크포인트가 있다면 삭제
        if best_checkpoint and os.path.exists(best_checkpoint):
            os.remove(best_checkpoint)
            # print(f" Removed old checkpoint: {best_checkpoint}")

        # 새로운 최적 체크포인트 경로 반환
        return checkpoint_path, accuracy

    return best_checkpoint, best_accuracy


def argparse_set():
    
    parser = argparse.ArgumentParser(description='endoscopy reconsruction') 

    parser.add_argument('--gpu', default='0')   
    parser.add_argument('--lr', default='0.0001', type=float)  
    parser.add_argument('--start_epoch', default=0, type=int)  
    parser.add_argument('--num_epochs', default=10, type=int)  
    parser.add_argument('--batch_size', default=64, type=int)  
    parser.add_argument('--seq_len', default=30, type=int)  
    parser.add_argument('--curtime', default='', type=str)  
    parser.add_argument('--root_dir', default='/data2/syupoh/dataset/endoscopy/videos/220209', type=str)  
    parser.add_argument('--annotation_file', default=f'220209_short_10_10.txt' , type=str)  
    parser.add_argument('--annotation_file2', default=f'', type=str)   
    parser.add_argument('--model_name', default=f'PSDeVCEM', type=str)  
    parser.add_argument('--data_mode', default=f'sup' , type=str)  
    parser.add_argument('--length_cut', default=1, type=float)  
    parser.add_argument('--train_ratio', default=0.8, type=float)  
    parser.add_argument('--ssl_ratio', default=0.5, type=float)  
    parser.add_argument('--sparsity_ratio', default=0.0, type=float)  
    parser.add_argument('--class_weights', default=0.3, type=float)  
    
    # parser.add_argument('--mode', default='UNet')   
    # parser.add_argument('--input', default='/data2/syupoh/dataset/endoscopy/videos/220209/annotations/220209_train_0_normal.txt')  

    parser.add_argument('--resume', default='')  
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test_code', action='store_true')


    args = parser.parse_args() 

    return args



def dataload_setting(args, root_dir, annotation_file, transform, mode, annotation_file2=''):

    if mode == 'mix':
        dataset_wce_sup = WCEVideoDataset(root_dir=root_dir, 
                                    annotation_file=annotation_file, 
                                    seq_len=args.seq_len,
                                    length_cut=args.length_cut, 
                                    transform=transform,
                                    mode = 'sup') # 'mil', 'sup'
        
        dataset_wce_mil = WCEVideoDataset(root_dir=root_dir, 
                                    annotation_file=annotation_file, 
                                    seq_len=args.seq_len,
                                    length_cut=args.length_cut, 
                                    transform=transform,
                                    mode = 'mil') # 'mil', 'sup'
        
    
        dataset_wce = ConcatDataset([dataset_wce_sup, dataset_wce_mil])

        train_size = int(args.train_ratio * len(dataset_wce))  # 80%를 훈련 데이터로 사용
        test_size = len(dataset_wce) - train_size  # 나머지를 테스트 데이터로 사용

        train_dataset_wce, test_dataset_wce = random_split(dataset_wce, [train_size, test_size])

    else:
        dataset_wce = WCEVideoDataset(root_dir=root_dir, 
                                    annotation_file=annotation_file, 
                                    seq_len=args.seq_len,
                                    length_cut=args.length_cut, 
                                    transform=transform,
                                    mode = mode) # 'mil', 'sup'
        
        if annotation_file2 == '':
            train_size = int(args.train_ratio * len(dataset_wce))  # 80%를 훈련 데이터로 사용
            test_size = len(dataset_wce) - train_size  # 나머지를 테스트 데이터로 사용

            train_dataset_wce, test_dataset_wce = random_split(dataset_wce, [train_size, test_size])
        else:
            test_dataset_wce = WCEVideoDataset(root_dir=root_dir, 
                                        annotation_file=annotation_file2, 
                                        seq_len=args.seq_len,
                                        length_cut=args.length_cut, 
                                        transform=transform,
                                        mode = mode) # 'mil', 'sup'
            train_dataset_wce = dataset_wce 
    
    
    print(f' train dataset {len(train_dataset_wce)}')
    print(f' test dataset {len(test_dataset_wce)}')

    img, label = train_dataset_wce[0]
    print(f"Image shape = {img.shape}")

    train_loader = DataLoader(train_dataset_wce, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset_wce, batch_size=args.batch_size, shuffle=False)


    return train_loader, test_loader


def train_loop(args, model, train_loader, test_loader):
    # 0.2 는 normal : abnormal = 3:1  positive weight -> [1.0 / 0.33] 
    # 0.3 는 normal : abnormal = 4:1  positive weight -> [1.0 / 0.25]
    # [(1/3) / (1/7)],

    # class 0, 1 -> 1 is positive (defect)
    # class weight torch.Size([30, 3, 224, 224])
    class_weights = torch.tensor([1.0 / args.class_weights], dtype=torch.float32).cuda() 
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) 

    scheduler = None
    if args.model_name == 'PSDeVCEM2':
        # Define the Adam optimizer
        optimizer = optim.Adam(
            model.parameters(),  # Model parameters to optimize
            lr=0.0001,           # Initial learning rate
            betas=(0.9, 0.999),  # Coefficients used for computing running averages of gradient and its square
            weight_decay=0.0001  # Weight decay (L2 penalty)
        )

        # Define the cyclic learning rate scheduler
        scheduler = CyclicLR(
            optimizer,
            base_lr=0.0001,       # Lower bound of the learning rate range
            max_lr=0.001,         # Upper bound of the learning rate range
            step_size_up=50,    # Number of training iterations to increase the learning rate
            mode='triangular',    # Learning rate policy (triangular2 is a good cyclic schedule)
            cycle_momentum=False,
        )

        # args.batch_size = 1
        # args.num_epochs = 500

    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    validation_interval = 1  # 매 1번째 에포크마다 검증

    val_loss = float("inf")
    best_accuracy = 0

    best_checkpoint = None
    if not args.resume == '':
        print(f' torch load {args.resume}')
        load_model = torch.load(args.resume)
        model.load_state_dict(
            load_model['model_state_dict']
            )
        optimizer.load_state_dict(load_model['optimizer_state_dict'] )

        if scheduler is not None:
            scheduler.load_state_dict(load_model['scheduler_state_dict'] )
        
        args.start_epoch = load_model['epoch']  
        best_accuracy = load_model['accuracy'] 

        best_checkpoint = args.resume
        args.model_name = args.resume.split('/')[1].split('_')[0]

    
        for key, value in load_model.items():
            print(f' {key} : {value} ', end="\n")
        # print("")

    accumulation_steps = 128

    ith = 0
    ssl = None
    sparsity_loss = None
    pbar_postfix = {}
    # 학습 루프
    for epoch in range(args.start_epoch, args.num_epochs):
        ith += 1
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader)
        pbar.set_description(f'Epoch - {epoch+1}')
        for videos, labels in pbar:
            videos = videos.cuda() # (batch_size num_Frames c w h) 
            labels = (labels > 0).float().cuda()
            labels = labels.unsqueeze(1)
                
            if args.batch_size > 1:
                optimizer.zero_grad()

            if args.model_name == 'PSDeVCEM2':
                outputs, attention_weights, ssl = model(videos) # v_outputs=outputs[0].item(), 
                loss = criterion(outputs, labels)
                loss = (1-args.ssl_ratio)*loss + args.ssl_ratio*ssl
                
                if args.sparsity_ratio > 0:
                    sparsity_loss = torch.mean(torch.abs(attention_weights))
                    loss = loss + args.sparsity_ratio * sparsity_loss

            elif args.model_name == 'STPN2':
                outputs, attention_weights, tcam = model(videos) # v_outputs=outputs[0].item(), 
                # loss = model.compute_loss(outputs, tcam, labels, attention_weights, beta=args.sparsity_ratio)
                        
                # 1. Classification loss based on T-CAM (segment-wise)
                loss = F.binary_cross_entropy(tcam.mean(dim=1), labels)

                # 2. Sparsity loss (L1 regularization on attention weights)
                sparsity_loss = torch.mean(torch.abs(attention_weights))
                
                # Total loss: weighted sum of classification loss and sparsity loss
                loss = loss + args.sparsity_ratio * sparsity_loss

            if args.batch_size == 1:
                loss /= accumulation_steps
            loss.backward()

            pbar_postfix["loss_bce"] = f"{loss.item():.04f}"

            if ssl is not None:
                pbar_postfix["loss_ssl"] = f"{ssl.item():.04f}"
            if sparsity_loss is not None:
                pbar_postfix["loss_sparsity"] = f"{sparsity_loss.item():.04f}"

            pbar.set_postfix(pbar_postfix)


            if args.batch_size == 1:
                loss 
                # Perform optimizer step every `accumulation_steps` batches
                if (ith + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()  # Update learning rate using the scheduler
            running_loss += loss.item()
            
        # Make sure that after the loop, any remaining gradients are applied
        if ith % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if epoch % validation_interval == 0:
            val_loss = 0
            
            with torch.no_grad():
                model.eval()
                correct = 0
                total = 0
                # 결과를 저장할 리스트 초기화
                all_preds = []
                all_labels = []
                for videos, labels in test_loader:
                    videos, labels = videos.cuda(), labels.cuda()
                    labels = (labels > 0).float().cuda()
                    labels = labels.unsqueeze(1)
                    
                    if args.model_name == 'PSDeVCEM2':
                        outputs, attention_weights, ssl = model(videos) # v_outputs=outputs[0].item(), 
                        loss = criterion(outputs, labels)
                        loss = (1-args.ssl_ratio)*loss + args.ssl_ratio*ssl

                        # Apply sigmoid to convert logits to probabilities for class 1
                        outputs = torch.sigmoid(outputs)  # Probabilities for class 1


                    elif args.model_name == 'STPN2':
                        outputs, attention_weights, tcam = model(videos) # v_outputs=outputs[0].item(), 
                        # loss = model.compute_loss(outputs, tcam, labels, attention_weights, beta=args.sparsity_ratio)
                        
                        # 1. Classification loss based on T-CAM (segment-wise)
                        tcam_classification_loss = F.binary_cross_entropy(tcam.mean(dim=1), labels)

                        # 2. Sparsity loss (L1 regularization on attention weights)
                        sparsity_loss = torch.mean(torch.abs(attention_weights))
                        
                        # Total loss: weighted sum of classification loss and sparsity loss
                        loss = tcam_classification_loss + args.sparsity_ratio * sparsity_loss

                        outputs = tcam.mean(dim=1)

                    val_loss += loss.item()


                    # Make class prediction (if probability > 0.5, predict 1; else predict 0)
                    pred = (outputs > 0.5).float()

                    # 정확도 계산
                    # _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
                    
                    # 예측값과 실제 레이블을 리스트에 저장
                    # predicted = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                    all_preds.extend(pred.cpu().numpy())  # GPU에서 CPU로 변환 후 numpy 배열로 변환
                    all_labels.extend(labels.cpu().numpy())  # GPU에서 CPU로 변환 후 numpy 배열로 변환
                    
                # Precision, Recall, F1 Score 계산
                precision = precision_score(all_labels, all_preds, average='binary')  # 이진 분류일 경우
                recall = recall_score(all_labels, all_preds, average='binary')
                f1 = f1_score(all_labels, all_preds, average='binary')

                val_loss /= len(test_loader)
                accuracy = correct / total * 100

                    
                print(f' val_loss : {val_loss:.4f} ' \
                      f'val_accuracy : {accuracy:2.2f} ' \
                      f'precision {precision:.4f} recall {recall:.4f} f1 {f1:.4f}'
                      )

                best_checkpoint, best_accuracy = save_best_checkpoint(args, epoch, model, 
                                                                    optimizer, val_loss, 
                                                                    best_accuracy, accuracy, scheduler,
                                                                    best_checkpoint=best_checkpoint)
                

    
    return model, best_accuracy


def train_process(args):    

    prev_name = f'log/{args.curtime}_{args.model_name}_gpu{args.gpu}.txt'
    f = open(prev_name, 'a')
    original_stdout = sys.stdout
    sys.stdout = DualOutput(f)  

    print(args)

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


#############################
#############################

    print('load model')
    start_time = time.time()  

    model = globals()[args.model_name]().cuda()
    model.apply(initialize_weights)

    print(f' -> {time.time()-start_time:.4f}s ')


    start_time = time.time()  

    print(f'load data {args.data_mode} ')
    train_loader, test_loader = dataload_setting(args, root_dir=args.root_dir, 
                                                            annotation_file=args.annotation_file, 
                                                            transform=transform, 
                                                            mode= args.data_mode,
                                                            annotation_file2=args.annotation_file2)

   
    print(f' -> {time.time()-start_time:.4f}s ')


#############################
#############################


    model, best_accuracy = train_loop(args, model, train_loader, test_loader)

    sys.stdout = original_stdout
    f.close()
    
    new_name = f'log/{args.curtime}_{args.model_name}_{best_accuracy:.2f}.txt'

    os.rename(prev_name, new_name)


def evaluation(args):

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model_name = os.path.basename(args.resume).split('.pth')[0]
    f = open(f'log/eval_{model_name}_gpu{args.gpu}.txt', 'a')
    original_stdout = sys.stdout
    sys.stdout = DualOutput(f)  
    
    print(f'==================================')
    print(args)

    print(f'load model {args.resume}')
    start_time = time.time()  
    

    model = globals()[args.model_name]().cuda()


    try:
        load_model = torch.load(args.resume)
        model.load_state_dict(
            load_model['model_state_dict']
            )

    except:
        print("no .pth file to read!")
        return 0

    print(f' -> {time.time()-start_time:.4f}s ')

    
    
    start_time = time.time()  

    print(f'load data {args.data_mode} ')
    _, test_loader = dataload_setting(args, root_dir=args.root_dir, 
                                                            annotation_file=args.annotation_file, 
                                                            transform=transform, 
                                                            mode= args.data_mode,
                                                            annotation_file2=args.annotation_file2)

   
    print(f' -> {time.time()-start_time:.4f}s ')



    accuracy = 0

    with torch.no_grad():
        model.eval()
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        pbar = tqdm(test_loader)
        for videos, labels in pbar:
            videos, labels = videos.cuda(), labels.cuda()
            labels = (labels > 0).float().cuda()
            # labels = labels.unsqueeze(1)

            total += labels.size(0)


            if args.model_name == 'PSDeVCEM2':
                outputs, attention_weights, ssl = model(videos) # v_outputs=outputs[0].item(), 

                # Apply sigmoid to convert logits to probabilities for class 1
                outputs = torch.sigmoid(outputs)  # Probabilities for class 1

            elif args.model_name == 'STPN2':
                outputs, attention_weights, tcam = model(videos) # v_outputs=outputs[0].item(), 

                outputs = tcam.mean(dim=1)

            # Make class prediction (if probability > 0.5, predict 1; else predict 0)
            pred = (outputs > 0.5).float()

            # # 정확도 계산
            correct += (pred == labels).sum().item()
            

            all_labels.extend(labels.cpu().numpy())  # GPU에서 CPU로 변환 후 numpy 배열로 변환
            all_preds.extend(pred.cpu().numpy())  # GPU에서 CPU로 변환 후 numpy 배열로 변환
            
            pbar.set_postfix({
                "pred" : pred.item(),
                "labels" : labels.item(),
                }
            )



        accuracy = correct / total * 100
    
        # Precision, Recall, F1 Score 계산
        precision = precision_score(all_labels, all_preds, average='binary')  # 이진 분류일 경우
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        print(f' accuracy : {accuracy:2.2f} precision {precision:.4f} recall {recall:.4f} f1 {f1:.4f}')

    sys.stdout = original_stdout
    f.close()
        
        
        
        
def test(args):
    args.gpu = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'

    batch_size = 32
    seq_len = 30
    root_dir = '/data2/syupoh/dataset/endoscopy/videos/220209'
    annotation_file = f'220209_short_10_10_100_0.2.txt' 
    args.model_name = 'STPN2'
    args.lr = 0.0005
    epoch = 1

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    print('load dataset')
    dataset_sup = WCEVideoDataset(root_dir=root_dir, 
                                  annotation_file=annotation_file, 
                                  seq_len=seq_len, 
                                  transform=transform,
                                  mode = 'sup')
    dataloader_sup = DataLoader(dataset_sup, batch_size=batch_size, shuffle=False)

    #########################
    #########################

    print('load model')
    model = globals()[args.model_name]().cuda()
    model.apply(initialize_weights)

    print(' model load complete')

    #########################
    #########################

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    pbar = tqdm(dataloader_sup)

    pbar.set_description(f'Epoch - {epoch+1}')
    for videos, labels in pbar:
        videos = videos.cuda() # (batch_size num_Frames c w h) 
        labels = (labels > 0).float().cuda()
        labels = labels.unsqueeze(1)
            

def main():
    args = argparse_set()

    now = datetime.now()
    curtime = now.isoformat()
    curtime = curtime.replace('-', '')
    curtime = curtime.replace(':', '')[2:13]
    args.curtime = curtime

    if args.test_code:
        test(args)


    torch.manual_seed(42)
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    
    if args.eval:
        evaluation(args)

        return 0
    
    train_process(args)


# Test the model with random input
if __name__ == "__main__":
    main()




