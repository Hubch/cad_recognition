import torch
import numpy as np
import argparse

from utils.datasets import get_data_loaders
from utils.models import get_model
from utils.matrices import evaluate_model


def val(device, model, val_loader, threshold):
    evaluate_model(device, model, data_loader, threshold_euc, threshold_cos)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = .0
        train_tqdm = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")
        
        for img1, img2, img3, label1, label2, label3 in train_tqdm:
            
            img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)
            anchor_label, positive_label, negative_label = label1.to(device), label2.to(device), label3.to(device)
            anchor_features, positive_features, negative_features = model(img1, img2, img3)

            optimizer.zero_grad()
        
            loss = criterion(anchor_features, positive_features, negative_features, anchor_label, positive_label, negative_label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            train_tqdm.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_epoch_loss}')

        # 更新学习率
        if scheduler:
            scheduler.step() # 使用余弦退火，不再需要参数
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 计算训练集上的阈值
        if epoch % 10 == 0:
            threshold_euc, threshold_cos = find_optimal_threshold(device, model, train_loader)
            print(f'Threshold found on train set: euclid = {threshold_euc}, cosine = {threshold_cos}')
            torch.save(model.state_dict(), weight_path)
            np.savez(threshold_path, [threshold_euc, threshold_cos])
        
        same_accuracy_euc, different_accuracy_euc, same_accuracy_cos, different_accuracy_cos = evaluate_model(device, model, val_loader, threshold_euc, threshold_cos)
        print(f'Validation Same Class Accuracy Euclid: {same_accuracy_euc}')
        print(f'Validation Different Class Accuracy Euclid: {different_accuracy_euc}')
        print(f'Validation Same Class Accuracy Cosine: {same_accuracy_cos}')
        print(f'Validation Different Class Accuracy Cosine: {different_accuracy_cos}')

if __name__ == '__main__':
# CLI arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--weight_path', type=str, help='where to save the trained weigths', default='./weights/cad_siamese.pth')
    parser.add_argument('--threshold_path', type=str, default='./weights/threshold.npz')
    parser.add_argument('--data', type=str, default='cropped_images')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = args.weight_path
    threshold_path = args.threshold_path
    data = args.data
    batch_size = args.batch_size
    
    thresholds = np.load(threshold_path)['arr_0']
    threshold_euc, threshold_cos = thresholds[0], thresholds[1]
    val_loader = get_data_loaders(root_dir=data, mode='val', batch_size=256)
    model = get_model(init=False, update=False, weight_path=weight_path).to(device)
    
    same_accuracy_euc, different_accuracy_euc, same_accuracy_cos, different_accuracy_cos = evaluate_model(device, model, train_loader, threshold_euc, threshold_cos)
    print(f'Validation Same Class Accuracy Euclid: {same_accuracy_euc}')
    print(f'Validation Different Class Accuracy Euclid: {different_accuracy_euc}')
    print(f'Validation Same Class Accuracy Cosine: {same_accuracy_cos}')
    print(f'Validation Different Class Accuracy Cosine: {different_accuracy_cos}')