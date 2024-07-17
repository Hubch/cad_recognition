import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from adabelief_pytorch import AdaBelief
from lookahead.optimizer import Lookahead

from tqdm import tqdm
import argparse

from utils.datasets import *
from utils.loss_functions import *
from utils.models import *
from utils.matrices import *

def train(device, model, criterion, optimizer, scheduler, num_epochs, train_loader, val_loader, weight_path, threshold_path):
    # torch.autograd.set_detect_anomaly(True)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = .0
        train_tqdm = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")

        for img1, img2, img3, label1, label2, label3 in train_tqdm:

            img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)
            anchor_label, positive_label, negative_label = label1.to(device), label2.to(device), label3.to(device)

            optimizer.zero_grad()

            with autocast():
                anchor_features = model(img1)
                positive_features = model(img2)
                negative_features = model(img3)
                loss = criterion(anchor_features, positive_features, negative_features, anchor_label, positive_label, negative_label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            train_tqdm.set_postfix(loss=loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_epoch_loss}')

        # 计算训练集上的阈值，更新 edge_index
        if epoch % 10 == 0:
            threshold_euc, threshold_cos = find_optimal_threshold(device, model, train_loader)
            print(f'Threshold found on train set: euclid = {threshold_euc}, cosine = {threshold_cos}')
            torch.save(model.state_dict(), weight_path)
            np.savez(threshold_path, [threshold_euc, threshold_cos])

        # 评估
        val_loss, same_accuracy_euc, different_accuracy_euc, same_accuracy_cos, different_accuracy_cos = evaluate_model(device, model, criterion, val_loader, threshold_euc, threshold_cos)
        print(f'Validation Same Class Accuracy Euclid: {same_accuracy_euc}')
        print(f'Validation Different Class Accuracy Euclid: {different_accuracy_euc}')
        print(f'Validation Same Class Accuracy Cosine: {same_accuracy_cos}')
        print(f'Validation Different Class Accuracy Cosine: {different_accuracy_cos}')

        # 更新学习率
        if scheduler:
            scheduler.step(val_loss) # 若使用余弦退火，不再需要参数

    torch.save(model.state_dict(), weight_path) # to save the whole model with architecture and weights

if __name__ == '__main__':
# CLI arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--init', action='store_true', help='train from default weight')
    parser.add_argument('--weight_path', type=str, help='where to save the trained weigths', default='./weights/cad_siamese.pth')
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

    # 初始化数据集、模型、损失函数和优化器
    train_loader = get_data_loaders(root_dir=data, mode='train', batch_size=batch_size)
    val_loader = get_data_loaders(root_dir=data, mode='val', batch_size=256)
    model = get_model(init=init, update=False, weight_path=weight_path).to(device)
    criterion = get_loss(margin)

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95, weight_decay=1e-2, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    optimizer = AdaBelief(model.parameters(), lr=learning_rate, weight_decay=1e-2, eps=1e-16, betas=(0.89,0.999), weight_decouple=True, rectify=True)
    # base_opt = optim.RAdam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    # base_opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    # optimizer = Lookahead(base_opt, k=5, alpha=0.5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=5, eta_min=1e-6)
    train(device, model, criterion, optimizer, scheduler, num_epochs, train_loader, val_loader, weight_path, threshold_path)