import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from adabelief_pytorch import AdaBelief
from lookahead.optimizer import Lookahead
from tqdm import tqdm
import argparse
from utils.datasets import get_data_loaders
from utils.loss_functions import get_loss, BYOL
from utils.models import get_teacher_model, get_student_model
from utils.matrices import find_optimal_threshold, evaluate_model
from torch.utils.tensorboard import SummaryWriter


def check_requires_grad(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name} does not require grad.")

def train_teacher(device, init, margin, learning_rate, num_class, num_epochs, train_loader, val_loader, weight_path, threshold_path):
    writter = SummaryWriter(log_dir="/root/tf-logs/teacher")
    
    model = get_teacher_model(num_classes=num_class, dropout_p=0.3, init=init, update=False, weight_path=weight_path, cuda=True).to(device)
    byol = BYOL(base_encoder=model).to(device)
    # byol = None
    
    criterion = get_loss(model=model, margin=margin, num_classes=num_class, num_epochs=num_epochs)
    
    base_optimizer = AdaBelief(model.parameters(), lr=learning_rate, weight_decay=1e-2, eps=1e-16, betas=(0.89, 0.999), weight_decouple=True, rectify=True)
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-8)
    
    scaler = GradScaler()
    best_val_loss = float('inf')
    best_score_cos = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        
        train_tqdm = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")
        for (anchor_img, contrastive_img, positive_img, negative_img, 
             anchor_label, positive_label, negative_label) in train_tqdm:
            
            anchor_img, contrastive_img, positive_img, negative_img = anchor_img.to(device), contrastive_img.to(device), positive_img.to(device), negative_img.to(device)
            anchor_label, positive_label, negative_label = anchor_label.to(device), positive_label.to(device), negative_label.to(device)
            
            optimizer.zero_grad()

            with autocast():
                anchor_features, anchor_logits = model(anchor_img)
                positive_features, positive_logits = model(positive_img)
                negative_features, negative_logits = model(negative_img)
                
                byol_loss = byol(anchor_img, contrastive_img)
                loss = criterion(anchor_features, positive_features, negative_features, 
                                 anchor_logits, positive_logits, negative_logits, 
                                 anchor_label, positive_label, negative_label, byol_loss )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            check_requires_grad(model)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            train_tqdm.set_postfix(loss=loss.item())

        if epoch % 10 == 0:
            threshold_euc, threshold_cos = find_optimal_threshold(device, model, train_loader)
            print(f'\nThreshold found on train set: euclid = {threshold_euc}, cosine = {threshold_cos}')
            np.savez(threshold_path+'_teacher.npz', [threshold_euc, threshold_cos])
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        torch.save(model.state_dict(), weight_path+"_teacher_last.pth")
        
        (
            val_loss, same_accuracy_euc, different_accuracy_euc, 
            same_class_accuracy_cos, different_class_accuracy_cos, classification_accuracy, 
            score_euc, score_cos
        ) = evaluate_model(device, model, criterion, val_loader, threshold_euc, threshold_cos)
        
        print(f'Epoch {epoch + 1}, Train Loss: {avg_epoch_loss}')
        print(f'Validation Loss:        {val_loss:.4f}')
        print(f'Validation Cls Acc:     {classification_accuracy:.4f}')
        print(f'Validation Pos Acc Euc: {same_accuracy_euc:.4f}')
        print(f'Validation Neg Acc Euc: {different_accuracy_euc:.4f}')
        print(f'Validation Pos Acc Cos: {same_class_accuracy_cos:.4f}')
        print(f'Validation Neg Acc Cos: {different_class_accuracy_cos:.4f}')
        print(f'Validation Score   Cos: {score_cos:.4f}')
        
        writter.add_scalar('Threshold/Euc', threshold_euc, epoch)
        writter.add_scalar('Threshold/Cos', threshold_cos, epoch)
        writter.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writter.add_scalar('Loss/Validation', val_loss, epoch)
        writter.add_scalar('Accuracy/Validation/Classification', classification_accuracy, epoch)
        writter.add_scalar('Accuracy/Validation/Same Class Euclid', same_accuracy_euc, epoch)
        writter.add_scalar('Accuracy/Validation/Different Class Euclid', different_accuracy_euc, epoch)
        writter.add_scalar('Accuracy/Validation/Same Class Cosine', same_class_accuracy_cos, epoch)
        writter.add_scalar('Accuracy/Validation/Different Class Cosine', different_class_accuracy_cos, epoch)
        writter.add_scalar('Score/Validation Euclid', score_euc, epoch)
        writter.add_scalar('Score/Validation Cosine', score_cos, epoch)


        if score_cos > best_score_cos:
            best_score_cos = score_cos
            torch.save(model.state_dict(), weight_path+'_teacher_best.pth')
            print(f'new best model saved into: {weight_path}_teacher_best.pth with acc={classification_accuracy:.4f}, score={score_cos:.4f}')
        
        byol.update_target_encoder()
        scheduler.step(val_loss)
    
    
    writter.close()


def train_student(device, init, margin, learning_rate, num_class, num_epochs, train_loader, val_loader, weight_path, threshold_path):
    writter = SummaryWriter(log_dir="/root/tf-logs/student")
    
    threshold_euc, threshold_cos = np.load(threshold_path+'_teacher.npz')['arr_0']
    print(f'\nThreshold generated by teacher: euc = {threshold_euc:.4f}, cosine = {threshold_cos:.4f}')
    
    model = get_student_model(num_classes=num_class, dropout_p=0.3, init=init, update=False, weight_path=weight_path, cuda=True).to(device)
    teacher = get_teacher_model(num_classes=num_class, dropout_p=0.3, init=False, update=False, weight_path=weight_path, cuda=True).to(device)
    
    criterion = get_loss(model=model, margin=margin, num_classes=num_class, num_epochs=None)
    
    base_optimizer = AdaBelief(model.parameters(), lr=learning_rate, weight_decay=1e-2, eps=1e-16, betas=(0.89, 0.999), weight_decouple=True, rectify=True)
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-8)
    
    scaler = GradScaler()
    best_val_loss = float('inf')
    best_score_cos = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        
        train_tqdm = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")
        for (anchor_img, contrastive_img, positive_img, negative_img, 
             anchor_label, positive_label, negative_label) in train_tqdm:
            
            anchor_img, contrastive_img, positive_img, negative_img = anchor_img.to(device), contrastive_img.to(device), positive_img.to(device), negative_img.to(device)
            anchor_labels, positive_labels, negative_labels = anchor_label.to(device), positive_label.to(device), negative_label.to(device)
            
            optimizer.zero_grad()

            with autocast():
                anchor_features, anchor_logits = model(anchor_img)
                positive_features, positive_logits = model(positive_img)
                negative_features, negative_logits = model(negative_img)
                teacher_features, teacher_logits = teacher(anchor_img)
                
                loss = criterion(anchor_features, positive_features, negative_features, 
                                 anchor_logits, positive_logits, negative_logits, 
                                 anchor_labels, positive_labels, negative_labels, 
                                 None, teacher_logits, teacher_features )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            check_requires_grad(model)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            train_tqdm.set_postfix(loss=loss.item())
        if epoch % 10 == 0:
            threshold_euc, threshold_cos = find_optimal_threshold(device, model, train_loader)
            print(f'\nThreshold found on train set: euclid = {threshold_euc}, cosine = {threshold_cos}')
            np.savez(threshold_path+'_student.npz', [threshold_euc, threshold_cos])
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        torch.save(model.state_dict(), weight_path+"_student_last.pth")
        
        (
            val_loss, same_accuracy_euc, different_accuracy_euc, 
            same_class_accuracy_cos, different_class_accuracy_cos, classification_accuracy, 
            score_euc, score_cos
        ) = evaluate_model(device, model, criterion, val_loader, threshold_euc, threshold_cos)
        
        print(f'Epoch {epoch + 1}, Train Loss: {avg_epoch_loss}')
        print(f'Validation Loss:        {val_loss:.4f}')
        print(f'Validation Cls Acc:     {classification_accuracy:.4f}')
        print(f'Validation Pos Acc Euc: {same_accuracy_euc:.4f}')
        print(f'Validation Neg Acc Euc: {different_accuracy_euc:.4f}')
        print(f'Validation Pos Acc Cos: {same_class_accuracy_cos:.4f}')
        print(f'Validation Neg Acc Cos: {different_class_accuracy_cos:.4f}')
        print(f'Validation Score   Cos: {score_cos:.4f}')
        
        writter.add_scalar('Threshold/Euc', threshold_euc, epoch)
        writter.add_scalar('Threshold/Cos', threshold_cos, epoch)
        writter.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writter.add_scalar('Loss/Validation', val_loss, epoch)
        writter.add_scalar('Accuracy/Validation/Classification', classification_accuracy, epoch)
        writter.add_scalar('Accuracy/Validation/Same Class Euclid', same_accuracy_euc, epoch)
        writter.add_scalar('Accuracy/Validation/Different Class Euclid', different_accuracy_euc, epoch)
        writter.add_scalar('Accuracy/Validation/Same Class Cosine', same_class_accuracy_cos, epoch)
        writter.add_scalar('Accuracy/Validation/Different Class Cosine', different_class_accuracy_cos, epoch)
        writter.add_scalar('Score/Validation Euclid', score_euc, epoch)
        writter.add_scalar('Score/Validation Cosine', score_cos, epoch)


        if score_cos > best_score_cos:
            best_score_cos = score_cos
            torch.save(model.state_dict(), weight_path+'_student_best.pth')
            print(f'new best model saved into: {weight_path}_student_best.pth with acc={classification_accuracy:.4f}, score={score_cos:.4f}')
        
        scheduler.step(val_loss)
    
    writter.close()
    
    

if __name__ == '__main__':
    # CLI arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--init', action='store_true', help='train from default weight')
    parser.add_argument('--weight_path', type=str, help='where to save the trained weights', default='./weights/cad_rep')
    parser.add_argument('--threshold_path', type=str, default='./weights/threshold.npz')
    parser.add_argument('--data', type=str, default='cropped_images')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01, help='init learning rate, 0.01 for SGD')
    parser.add_argument('--margin', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init = args.init
    weight_path = args.weight_path
    threshold_path = args.threshold_path
    data = args.data
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    margin = args.margin

    train_loader, num_class = get_data_loaders(root_dir=data, mode='train', batch_size=batch_size)
    val_loader, _ = get_data_loaders(root_dir=data, mode='val', batch_size=batch_size)
    
    # train_teacher(device, init, margin, learning_rate, num_class, num_epochs, train_loader, val_loader, weight_path, threshold_path)
    
    train_student(device, init, margin, learning_rate, num_class, num_epochs, train_loader, val_loader, weight_path, threshold_path)