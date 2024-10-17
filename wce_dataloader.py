import torch
import torch.nn as nn
import torchvision.models as models

import os
from PIL import Image

from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset
from torchvision import transforms

import pdb
import time
from tqdm import tqdm


class WCEVideoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, seq_len=10, length_cut=1, transform=None, mode='mil'):
        self.root_dir = root_dir
        self.transform = transform
        self.image_path_set = []
        self.annotations = []
        self.seq_len = seq_len


        annotation_file = f'{root_dir}/annotations/{annotation_file}' 

        try:
            with open(annotation_file, 'r') as f:
                line_set = f.readlines()
        except:
            with open(annotation_file, 'r', encoding='utf-16') as f:
                line_set = f.readlines()
        
        if mode == 'sup':
                
            self.count = 0 # anomal data
            for ith in range(len(line_set)):
                path1, length1, label1 = line_set[ith].strip().split()
                name1 = "".join(path1.split('/')[1].split('_')[0:2])

                file1 = sorted(os.listdir(os.path.join(self.root_dir, path1)))

                length1 = int(length1)
                if length1 > seq_len * 10:
                    length1 = length1 * length_cut # 하나의 segment 가 너무 길면 잘라버리기 1이면 안자른다. (default = 1)
                    
                length1 = int(length1)
                for seq_number in range(length1 // seq_len):

                    image_path_set_temp = [os.path.join(self.root_dir, path1, item) for item in file1[seq_number*seq_len:(seq_number+1)*seq_len]]
                    
                    if len(image_path_set_temp) == seq_len:
                        self.image_path_set.append(image_path_set_temp)
                        self.annotations.append(label1)

                        if not label1 == '0':
                            self.count += 1

        elif mode == 'mil':

            self.count = 0 # middle anomaly
            self.count1 = 0 # case1
            self.count2 = 0 # case2
            self.count3 = 0 # case3

            for ith in range(len(line_set)):
                if ith == len(line_set) - 2:
                    break

                path1, length1, label1 = line_set[ith].strip().split()
                name1 = "".join(path1.split('/')[1].split('_')[0:2])

                path2, length2, label2 = line_set[ith+1].strip().split()
                name2 = "".join(path2.split('/')[1].split('_')[0:2])
                
                path3, length3, label3 = line_set[ith+2].strip().split()
                name3 = "".join(path3.split('/')[1].split('_')[0:2])

                if label2 == "0":
                    continue

                if not label2 == "0":
                    self.count += 1

                file1 = sorted(os.listdir(os.path.join(self.root_dir, path1)))
                file2 = sorted(os.listdir(os.path.join(self.root_dir, path2)))
                file3 = sorted(os.listdir(os.path.join(self.root_dir, path3)))


                length1 = int(length1)
                length2 = int(length2)
                length3 = int(length3)

                if length2 < seq_len:
                    # case1 # eg) length2 = 17

                    self.count1 += 1
                    left = (seq_len - length2) // 2 
                    right = left

                    # 홀수면 하나 더해줘
                    if length2 % 2 > 0:
                        right += 1
                    
                    image_path_set_temp = [os.path.join(self.root_dir, path1, item) for item in file1[-left:]]
                    image_path_set_temp.extend([os.path.join(self.root_dir, path2, item) for item in file2])
                    image_path_set_temp.extend([os.path.join(self.root_dir, path3, item) for item in file3[:right]])

                    if label1 != "0":
                        annotations_temp = label1
                    elif label2 != "0":
                        annotations_temp = label2
                    elif label3 != "0":
                        annotations_temp = label3
                    else:
                        annotations_temp = "0"  

                    if len(image_path_set_temp) == seq_len:
                        self.image_path_set.append(image_path_set_temp)
                        self.annotations.append(annotations_temp)
                    else:
                        print(length1)
                        print(length2)
                        print(length3)
                        pdb.set_trace()


                elif seq_len < length2 and length2 < seq_len * 2:
                    # case2 # eg) length2 = 47

                    self.count2 += 1
                    half_seq_len = int(seq_len // 2) # 15
                    
                    # length // 2 -> 23
                    left = int(length2 // 2) - half_seq_len  # 8 
                    right = int(length2 // 2) + half_seq_len  # 38


                    ### left 
                    image_path_set_temp = [os.path.join(self.root_dir, path1, item) for item in file1[left-seq_len:]]
                    image_path_set_temp.extend([os.path.join(self.root_dir, path2, item) for item in file2[:left]])                
                    
                    if label1 != "0":
                        annotations_temp = label1
                    elif label2 != "0":
                        annotations_temp = label2
                    else:
                        annotations_temp = "0" 

                    if not len(image_path_set_temp) == seq_len:
    
                        image_path_set_temp = [os.path.join(self.root_dir, path1, item) for item in file1]
                        image_path_set_temp.extend([os.path.join(self.root_dir, path2, item) for item in file2[:seq_len - length1]])   

                        # print(len(image_path_set_temp))
                        # print(f'length1 {length1}')
                        # print(f'length2 {length2}')
                        
                    self.image_path_set.append(image_path_set_temp)
                    self.annotations.append(annotations_temp)


                    ## ## center
                    # image_path_set_temp = [os.path.join(self.root_dir, path2, item) for item in file2[left:left+seq_len]]

                    # if len(image_path_set_temp) == seq_len:
                    #     self.image_path_set.append(image_path_set_temp)
                    #     self.annotations.append(label2)


                    ### right
                    image_path_set_temp = [os.path.join(self.root_dir, path2, item) for item in file2[right:]]
                    image_path_set_temp.extend([os.path.join(self.root_dir, path3, item) for item in file3[:seq_len - (length2 - right)]])
                                
                    if label2 != "0":
                        annotations_temp = label2
                    elif label3 != "0":
                        annotations_temp = label3
                    else:
                        annotations_temp = "0"  
                        
                    if not len(image_path_set_temp) == seq_len:

                        image_path_set_temp = [os.path.join(self.root_dir, path2, item) for item in file2[-(seq_len - length3):]]
                        image_path_set_temp.extend([os.path.join(self.root_dir, path3, item) for item in file3])  
                        
                        # print(len(image_path_set_temp)) 
                        # print(f'length2 {length2}')
                        # print(f'length3 {length3}')

                    self.image_path_set.append(image_path_set_temp)
                    self.annotations.append(annotations_temp)

                
                else:
                    # case3 # eg) length2 = 372

                    self.count3 += 1
                    half_seq_len = int(seq_len // 2) # 15
                    half = length2 // 2 # 186
                    left = half - half_seq_len # 171
                    left = (left % seq_len) # 21 -> 20
                    right = length2 - left # 350

                    ### left 
                    if label1 != "0":
                        annotations_temp = label1
                    elif label2 != "0":
                        annotations_temp = label2
                    else:
                        annotations_temp = "0"  

                    image_path_set_temp = [os.path.join(self.root_dir, path1, item) for item in file1[left-seq_len:]]
                    image_path_set_temp.extend([os.path.join(self.root_dir, path2, item) for item in file2[:left]])
                                    

                    if not len(image_path_set_temp) == seq_len:
                        image_path_set_temp = [os.path.join(self.root_dir, path1, item) for item in file1]
                        image_path_set_temp.extend([os.path.join(self.root_dir, path2, item) for item in file2[:seq_len - length1]])   
                        
                        # print(len(image_path_set_temp))
                        # print(f'length1 {length1}')
                        # print(f'length2 {length2}')
                        
                    self.image_path_set.append(image_path_set_temp)
                    self.annotations.append(annotations_temp)

    
                    # ### center
                    # for ith in range( (right - left) // seq_len):

                    #     image_path_set_temp = [os.path.join(self.root_dir, path2, item) for item in file2[left+ith*seq_len:left+(ith+1)*seq_len]]
                        
                    #     if len(image_path_set_temp) == seq_len:
                    #         self.image_path_set.append(image_path_set_temp)
                    #         self.annotations.append(label2)


                    ### right
                    if label2 != "0":
                        annotations_temp = label2
                    elif label3 != "0":
                        annotations_temp = label3
                    else:
                        annotations_temp = "0"  

                    
                    image_path_set_temp = [os.path.join(self.root_dir, path2, item) for item in file2[right:]]
                    image_path_set_temp.extend([os.path.join(self.root_dir, path3, item) for item in file3[:seq_len - (length2 - right)]])
                                    
                    if not len(image_path_set_temp) == seq_len:

                        image_path_set_temp = [os.path.join(self.root_dir, path2, item) for item in file2[-(seq_len - length3):]]
                        image_path_set_temp.extend([os.path.join(self.root_dir, path3, item) for item in file3])   
                        
                        # print(len(image_path_set_temp))
                        # print(f'length2 {length2}')
                        # print(f'length3 {length3}')
                    

                    self.image_path_set.append(image_path_set_temp)
                    self.annotations.append(annotations_temp)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # (batch_size num_Frames c w h)
        image_paths = self.image_path_set[idx]
        annotations = int(self.annotations[idx])


        image_list = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            image_list.append(image.unsqueeze(0))
        
        label = torch.tensor(annotations)
        input_data = torch.cat(image_list, dim=0)

        
        return input_data, label
    



def main():
    print('WCE loader script')
    
    batch_size = 32
    seq_len = 30
    root_dir = '/data2/syupoh/dataset/endoscopy/videos/220209'
    annotation_file = f'{root_dir}/annotations/220209_short_10_10.txt' 

    
    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print('load dataset')
    # dataset = WCEVideoDataset(root_dir=root_dir, annotation_file=annotation_file, transform=transform)
    dataset_mil = WCEVideoDataset(root_dir=root_dir, 
                                  annotation_file=annotation_file, 
                                  seq_len=seq_len, 
                                  transform=transform,
                                  mode = 'mil')
    dataloader_mil = DataLoader(dataset_mil, batch_size=batch_size, shuffle=False)

    
    # print(f' middle anomaly {dataset_mil.count}')
    # print(f' case1 {dataset_mil.count1}')
    # print(f' case2 {dataset_mil.count2}')
    # print(f' case3 {dataset_mil.count3}')

    # print(f' expect {dataset_mil.count1+dataset_mil.count2*2+dataset_mil.count3*2}')

    # dataset_sup = WCEVideoDataset(root_dir=root_dir, 
    #                               annotation_file=annotation_file, 
    #                               seq_len=seq_len, 
    #                               transform=transform,
    #                               mode = 'sup')
    # dataloader_sup = DataLoader(dataset_sup, batch_size=batch_size, shuffle=False)

    # print(f' sup full set {len(dataset_sup)}')
    # print(f' sup anomal {dataset_sup.count}')
    # print(len(dataset_mil))


    print(f'batch size : {batch_size}')
    for ith, (image, label) in enumerate(dataloader_mil):
        print(f' {ith+1} / {len(dataloader_mil)}')
        # print(image.shape)
        # print(label.shape)
        # pdb.set_trace()


    
# Test the model with random input
if __name__ == "__main__":
    main()




