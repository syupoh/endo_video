
import os
import sys

import torch
import argparse


class DualOutput:
    def __init__(self, file):
        self.file = file
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        # This flush method is needed for Python 3 compatibility.
        # This handles the flush command by doing nothing.
        # You might want to specify some extra behavior here.
        self.terminal.flush()
        self.file.flush()


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
    parser.add_argument('--annotation_file2', default=f'' , type=str)   
    parser.add_argument('--model_name', default=f'PSDeVCEM' , type=str)  
    parser.add_argument('--data_mode', default=f'sup' , type=str)  
    parser.add_argument('--length_cut', default=1, type=float)  
    parser.add_argument('--train_ratio', default=0.8, type=float)  
    parser.add_argument('--ssl_ratio', default=0.5, type=float)  
    parser.add_argument('--sparsity_ratio', default=0.1, type=float)  
    parser.add_argument('--class_weights', default=0.3, type=float)  
    
    # parser.add_argument('--mode', default='UNet')   
    # parser.add_argument('--input', default='/data2/syupoh/dataset/endoscopy/videos/220209/annotations/220209_train_0_normal.txt')  

    parser.add_argument('--resume', default='')  
    parser.add_argument('--ssl_loss', action='store_true')  
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test_code', action='store_true')


    args = parser.parse_args() 

    return args


# 체크포인트 저장 함수 (최적의 모델만 저장하고, 이전 체크포인트 삭제)
def save_best_checkpoint(args, epoch, model, optimizer, 
                         val_loss, best_accuracy, accuracy, checkpoint_dir="checkpoints", best_checkpoint=None):
    
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
        }, checkpoint_path)

        print(f" New best model saved at {checkpoint_path}")

        # 이전 최적의 체크포인트가 있다면 삭제
        if best_checkpoint and os.path.exists(best_checkpoint):
            os.remove(best_checkpoint)
            # print(f" Removed old checkpoint: {best_checkpoint}")

        # 새로운 최적 체크포인트 경로 반환
        return checkpoint_path, accuracy

    return best_checkpoint, best_accuracy

def main():
    pass


# Test the model with random input
if __name__ == "__main__":
    main()



