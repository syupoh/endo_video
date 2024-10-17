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

from wce_utils import DualOutput, save_best_checkpoint, argparse_set
from wce_model import PSDeVCEM, STPN, PSDeVCEM2
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
    # 모델, 손실 함수, 옵티마이저 정의

    # 0.2 는 normal : abnormal = 3:1  positive weight -> 1.0 / 0.33 
    # 0.3 는 normal : abnormal = 4:1  positive weight -> 1.0 / 0.25 
    #

    class_weights = torch.tensor([1.0 / args.class_weights], dtype=torch.float32).cuda() 
    # if '0.2' in args.annotation_file:
    #     class_weights = torch.tensor([1.0 / 0.33], dtype=torch.float32).cuda() 
    # elif '0.3' in args.annotation_file:
    #     class_weights = torch.tensor([1.0 / 0.25], dtype=torch.float32).cuda() 
    # else:
    #     class_weights = torch.tensor([(1/3) / (1/7)], dtype=torch.float32).cuda() 
    
    # class 0, 1 -> 1 is positive (defect)
    # class weight torch.Size([30, 3, 224, 224])
     
    criterion2 = nn.BCEWithLogitsLoss() 
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) 

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

    # f = open(f'log/{args.curtime}_gpu{args.gpu}.txt', 'a')
    # original_stdout = sys.stdout
    # sys.stdout = DualOutput(f)  

    best_loss = float("inf")
    best_checkpoint = None
    best_accuracy = 0
    if not args.resume == '':
        print(f' torch load {args.resume}')
        load_model = torch.load(args.resume)
        model.load_state_dict(
            load_model['model_state_dict']
            )
        optimizer.load_state_dict(load_model['optimizer_state_dict'] )
        args.start_epoch = load_model['epoch']  

        best_accuracy = load_model['accuracy'] 
        best_checkpoint = args.resume
        args.model_name = args.resume.split('/')[1].split('_')[0]

    try:
        val_loss
        accuracy
    except NameError:
        val_loss = 0
        accuracy = 0

    accumulation_steps = 128

    ith = 0
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


            if args.model_name == 'PSDeVCEM':
                outputs, attn_weights, ssl = model(videos) # v_outputs=outputs[0].item(), 
            elif args.model_name == 'PSDeVCEM2':
                outputs, ssl = model(videos) # v_outputs=outputs[0].item(), 
            elif args.model_name == 'STPN2':
                outputs, attention_weights, tcam, sparsity_loss = model(videos) # v_outputs=outputs[0].item(), 

            loss = criterion(outputs, labels)

            # loss1 = criterion(outputs[5], labels[5])
            # loss2 = criterion2(outputs[5], labels[5])

            # for i in range(args.batch_size):
            #     print(f' {outputs[i].item():.4f} {labels[i].item()} ')
            #     print(f' classweight {loss1.item():.4f} general {loss2.item():.4f} ')
            #     pdb.set_trace()

            # print(f' {outputs[0].item():.4f} {labels[0].item()} ')
            # print(f' classweight {loss1.item():.4f} general {loss2.item():.4f} ')
            # pdb.set_trace()
        ##################
        #             """
        # Compute the total loss (classification + sparsity + T-CAM influence)
        # """
        # # 1. Classification loss based on T-CAM (segment-wise)
        # tcam_classification_loss = F.binary_cross_entropy(tcam.mean(dim=1), target)

        # # 2. Sparsity loss (L1 regularization on attention weights)
        # sparsity_loss = torch.mean(torch.abs(attention_weights))
        
        # # Total loss: weighted sum of classification loss and sparsity loss
        # total_loss = tcam_classification_loss + beta * sparsity_loss
        ##################

            if args.ssl_loss:
                loss = (1-args.ssl_ratio)*loss + args.ssl_ratio*ssl
                
            if args.sparsity_loss:
                loss = (1-args.sparsity_ratio)*loss + args.sparsity_ratio*sparsity_loss

            loss.backward()

            if args.model_name == 'PSDeVCEM':
                pbar.set_postfix({
                    "loss_bce" : loss.item(), 
                    "loss_ssl" : ssl.item(),
                    "val_loss" : val_loss,
                    'val_acc': f"{accuracy:2.2f}"
                    }
                )
            elif args.model_name == 'PSDeVCEM2':
                pbar.set_postfix({
                    "loss_bce" : loss.item(), 
                    "loss_ssl" : ssl.item(),
                    "val_loss" : val_loss,
                    'val_acc': f"{accuracy:2.2f}"
                    }
                )
            elif args.model_name == 'STPN2':
                pbar.set_postfix({
                    "loss_bce" : loss.item(), 
                    "val_loss" : val_loss,
                    'val_acc': f"{accuracy:2.2f}"
                    }
                )
            
            if args.batch_size == 1:
                # Perform optimizer step every `accumulation_steps` batches
                if (ith + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()  # Update learning rate using the scheduler
            running_loss += loss.item()
            

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
                    
                    if args.model_name == 'PSDeVCEM':
                        outputs, attn_weights, ssl = model(videos) # v_outputs=outputs[0].item(), 
                    elif args.model_name == 'PSDeVCEM2':
                        outputs, ssl = model(videos) # v_outputs=outputs[0].item(), 
                    elif args.model_name == 'STPN2':
                        class_scores, attention_weights, tcam, sparsity_loss = model(videos) # v_outputs=outputs[0].item(), 
                    
                    loss = criterion(outputs, labels)
        
                    
                    if args.ssl_loss:
                        loss = loss + ssl
                    val_loss += loss.item()


                    # Apply sigmoid to convert logits to probabilities for class 1
                    probs = torch.sigmoid(outputs)  # Probabilities for class 1

                    # Make class prediction (if probability > 0.5, predict 1; else predict 0)
                    pred = (probs > 0.5).float()

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

                if args.model_name == 'PSDeVCEM':
                    pbar.set_postfix({
                        "loss_bce" : loss.item(), 
                        "loss_ssl" : ssl.item(),
                        "val_loss" : val_loss,
                        'val_acc': f"{accuracy:2.2f}"
                        }
                    )
                elif args.model_name == 'PSDeVCEM2':
                    pbar.set_postfix({
                        "loss_bce" : loss.item(), 
                        "loss_ssl" : ssl.item(),
                        "val_loss" : val_loss,
                        'val_acc': f"{accuracy:2.2f}"
                        }
                    )
                elif args.model_name == 'STPN2':
                    pbar.set_postfix({
                        "loss_bce" : loss.item(), 
                        "val_loss" : val_loss,
                        'val_acc': f"{accuracy:2.2f}"
                        }
                    )
                    
                print(f' val_loss : {val_loss:.4f} ' \
                      f'accuracy : {accuracy:2.2f} ' \
                      f'precision {precision:.4f} recall {recall:.4f} f1 {f1:.4f}'
                      )

                # # Save the model if validation loss improves
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     torch.save(model.state_dict(), 'best_model.pth')

                # # Early stopping based on validation error (optional)
                # if val_loss == best_val_loss:
                #     print(f"Early stopping at epoch {epoch+1} due to lowest validation loss.")
                #     break

                best_checkpoint, best_accuracy = save_best_checkpoint(args, epoch, model, 
                                                                    optimizer, val_loss, 
                                                                    best_accuracy, accuracy,
                                                                    best_checkpoint=best_checkpoint)
                

            # print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader_mil)}")
   
    # sys.stdout = original_stdout
    # f.close()
    
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

            if args.model_name == 'PSDeVCEM':
                outputs, attn_weights, ssl = model(videos) # v_outputs=outputs[0].item(), 
            elif args.model_name == 'PSDeVCEM2':
                outputs, ssl = model(videos) # v_outputs=outputs[0].item(), 
            elif args.model_name == 'STPN2':
                class_scores, attention_weights, tcam, sparsity_loss = model(videos) # v_outputs=outputs[0].item(), 
            
            
            # Apply sigmoid to convert logits to probabilities for class 1
            probs = torch.sigmoid(outputs)  # Probabilities for class 1

            # Make class prediction (if probability > 0.5, predict 1; else predict 0)
            pred = (probs > 0.5).float()

            # # 정확도 계산
            # _, predicted = torch.max(outputs, 1)
            # pred = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            correct += (pred == labels).sum().item()
            

            all_labels.extend(labels.cpu().numpy())  # GPU에서 CPU로 변환 후 numpy 배열로 변환
            all_preds.extend(pred.cpu().numpy())  # GPU에서 CPU로 변환 후 numpy 배열로 변환
            
            pbar.set_postfix({
                # "predicted" : predicted.item(), 
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

    # Example usage
    batch_size = 8
    num_segments = 30
    feature_dim = 1024  # Feature dimension for each segment
    num_classes = 2   # Number of action classes

    # Randomly generated features (RGB and flow)
    rgb_features = torch.randn(batch_size, num_segments, feature_dim)
    flow_features = torch.randn(batch_size, num_segments, feature_dim)

    #########################
    #########################

    print('load model')
    model = globals()[args.model_name]().cuda()
    model.apply(initialize_weights)

    print(' model load complete')

    #########################
    #########################

    
    #########################
    #########################
        # # class 0, 1 -> 1 is positive (defect)
        # # class weight torch.Size([30, 3, 224, 224])
    class_weights1 = torch.tensor([1.0 / 0.33], dtype=torch.float32).cuda() 
    class_weights2 = torch.tensor([0.33], dtype=torch.float32).cuda() 
    class_weights3 = torch.tensor([1.0 / 0.005], dtype=torch.float32).cuda() 

    criterion = nn.BCEWithLogitsLoss() 
    criterion1 = nn.BCEWithLogitsLoss(pos_weight=class_weights1) 
    criterion2 = nn.BCEWithLogitsLoss(pos_weight=class_weights2) 
    criterion3 = nn.BCEWithLogitsLoss(pos_weight=class_weights3)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    
    # # model.train()
    # model.eval()
    # with torch.no_grad():
    #     for image, label in dataloader_sup:


    #         start_time = time.time()  
    #         image = image.cuda()


    #         # optimizer.zero_grad()
            
    #         # if (label == 1).any().item():
    #         #     print(f"{label}")
    #         label = (label > 0).float().cuda()
    #         label = label.unsqueeze(1)
            
    #         if args.model_name == 'PSDeVCEM2':
    #             output, ssl = model(image)
    #         elif args.model_name == 'PSDeVCEM':
    #             output, attn_weights, ssl = model(image)
    #         elif args.model_name == 'STPN2':
    #             output, attn_weights= model(image)
                
            
    #         ##############
    #         ##############
    #         # print(f' output {output.shape}')
    #         # print(f' attn_weights {attn_weights.shape}')
    #         # print(f' ssl {ssl.shape}')


    #         # Apply sigmoid to convert logits to probabilities for class 1
    #         probs = torch.sigmoid(output)  # Probabilities for class 1

    #         # Make class prediction (if probability > 0.5, predict 1; else predict 0)
    #         pred = (probs > 0.5).float()
    #         print(f' output {output[:,0]}, \n label {label[:,0]}\n pred {pred[:,0]}')


    #         # loss = criterion(output, label)
    #         # print(f'loss {loss:.4f}\n')

    #         # loss.backward()
    #         # optimizer.step()


    #         # ##############
    #         # ##############
            


    #         # loss = criterion(output, label)
    #         # loss1 = criterion1(output, label)
    #         # loss2 = criterion2(output, label)
    #         # loss3 = criterion3(output, label)
    #         # # print(image.shape)
    #         # # print(label.shape)

    #         # # loss3.backward()
    #         # # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         # # optimizer.step()

    #         # print(f' output {output[:,0]}, \n label {label[:,0]}\n pred {pred[:,0]}\n' +
    #         #       f' loss {loss.item():.4f} loss1 {loss1.item():.4f} loss2 {loss2.item():.4f} loss3 {loss3.item():.4f}')

    #         execution_time = time.time() - start_time  # 실행 시간 계산
    #         print(f"Execution time: {execution_time:.6f} seconds")
    #         start_time = time.time()  

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




