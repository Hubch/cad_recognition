import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_histogram(positive_distances, negative_distances, title):
    plt.figure(figsize=(10, 5))
    plt.hist(positive_distances, bins=50, alpha=0.6, label='Positive Pairs')
    plt.hist(negative_distances, bins=50, alpha=0.6, label='Negative Pairs')
    plt.title(f'Distribution of {title}')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'/root/tf-logs/teacher/latest_distance_histogram_{title}.png')

def compute_distances(device, model, data_loader):
    model.eval()
    positive_distances_euclid = []
    negative_distances_euclid = []
    positive_distances_cosine = []
    negative_distances_cosine = []

    with torch.no_grad():
        for anchor_img, contrastive_img, positive_img, negative_img, anchor_label, positive_label, negative_label in data_loader:
            anchor_img, positive_img, negative_img = anchor_img.to(device), positive_img.to(device), negative_img.to(device)

            anchor_feature, _ = model(anchor_img)
            positive_feature, _ = model(positive_img)
            negative_feature, _ = model(negative_img)

            positive_distance_euclid = F.pairwise_distance(anchor_feature, positive_feature).cpu().numpy()
            negative_distance_euclid = F.pairwise_distance(anchor_feature, negative_feature).cpu().numpy()
            positive_distance_cosine = 1 - F.cosine_similarity(anchor_feature, positive_feature).cpu().numpy()
            negative_distance_cosine = 1 - F.cosine_similarity(anchor_feature, negative_feature).cpu().numpy()
            positive_distances_euclid.extend(positive_distance_euclid)
            negative_distances_euclid.extend(negative_distance_euclid)
            positive_distances_cosine.extend(positive_distance_cosine)
            negative_distances_cosine.extend(negative_distance_cosine)

    plot_histogram(positive_distances_euclid, negative_distances_euclid, "Euclidean Distance")
    plot_histogram(positive_distances_cosine, negative_distances_cosine, "Cosine Distance")
    return np.array(positive_distances_euclid), np.array(negative_distances_euclid), \
           np.array(positive_distances_cosine), np.array(negative_distances_cosine)


def find_optimal_threshold(device, model, data_loader):
    positive_distances_euc, negative_distances_euc, \
    positive_distances_cos, negative_distances_cos = \
    compute_distances(device, model, data_loader)

    # 计算正样本和负样本距离的均值和标准差
    positive_mean_euc = np.mean(positive_distances_euc)
    negative_mean_euc = np.mean(negative_distances_euc)
    positive_std_euc = np.std(positive_distances_euc)
    negative_std_euc = np.std(negative_distances_euc)

    positive_mean_cos = np.mean(positive_distances_cos)
    negative_mean_cos = np.mean(negative_distances_cos)
    positive_std_cos = np.std(positive_distances_cos)
    negative_std_cos = np.std(negative_distances_cos)
    
    print(f"[debug] pos_mean_euc: {positive_mean_euc:.4f}")
    print(f"[debug] neg_mean_euc: {negative_mean_euc:.4f}")
    print(f"[debug] pos_mean_cos: {positive_mean_cos:.4f}")
    print(f"[debug] neg_mean_cos: {negative_mean_cos:.4f}")

    # 使用均值和标准差来确定阈值范围
    lower_bound_euc = positive_mean_euc + positive_std_euc
    upper_bound_euc = negative_mean_euc - negative_std_euc

    lower_bound_cos = positive_mean_cos + positive_std_cos
    upper_bound_cos = negative_mean_cos - negative_std_cos

    # 设置步长和范围
    steps = 100
    step_size_euc = (upper_bound_euc - lower_bound_euc) / steps
    step_size_cos = (upper_bound_cos - lower_bound_cos) / steps

    best_threshold_euc = lower_bound_euc
    best_threshold_cos = lower_bound_cos
    best_euc = 0
    best_cos = 0

    for i in range(steps):
        threshold_euc = lower_bound_euc + i * step_size_euc
        tp_euc = np.sum(positive_distances_euc < threshold_euc)
        fn_euc = np.sum(positive_distances_euc > threshold_euc)
        tn_euc = np.sum(negative_distances_euc > threshold_euc)
        fp_euc = np.sum(negative_distances_euc < threshold_euc)

        precision_euc = tp_euc / (tp_euc + fn_euc + 1e-8)
        recall_euc = tn_euc / (tn_euc + fp_euc + 1e-8)
        score_euc = 2 * (precision_euc * recall_euc) / (precision_euc + recall_euc + 1e-8)
        # score_euc = (tp_euc + tn_euc) / (len(positive_distances_euc) + len(negative_distances_euc))

        threshold_cos = lower_bound_cos + i * step_size_cos
        tp_cos = np.sum(positive_distances_cos < threshold_cos)
        fn_cos = np.sum(positive_distances_cos > threshold_cos)
        tn_cos = np.sum(negative_distances_cos > threshold_cos)
        fp_cos = np.sum(negative_distances_cos < threshold_cos)

        precision_cos = tp_cos / (tp_cos + fn_cos + 1e-8)
        recall_cos = tn_cos / (tn_cos + fp_cos + 1e-8)
        score_cos = 2 * (precision_cos * recall_cos) / (precision_cos + recall_cos + 1e-8)
        # score_cos = (tp_cos + tn_cos) / (len(positive_distances_cos) + len(negative_distances_cos))

        if score_euc > best_euc:
            best_euc = score_euc
            best_threshold_euc = threshold_euc

        if score_cos > best_cos:
            best_cos = score_cos
            best_threshold_cos = threshold_cos

    return best_threshold_euc, best_threshold_cos

def evaluate_model(device, model, criterion, val_loader, threshold_euc, threshold_cos):
    model.eval()
    val_loss = 0.0
    all_class_preds = []
    all_class_targets = []

    positive_distances_euc = []
    negative_distances_euc = []
    positive_distances_cos = []
    negative_distances_cos = []

    with torch.no_grad():
        for (anchor_img, contrastive_img, positive_img, negative_img, anchor_label, positive_label, negative_label) in tqdm(val_loader, desc='Evaluating'):
            anchor_img, contrastive_img, positive_img, negative_img = anchor_img.to(device), contrastive_img.to(device), positive_img.to(device), negative_img.to(device)
            anchor_label, positive_label, negative_label = anchor_label.to(device), positive_label.to(device), negative_label.to(device)
            
            anchor_features, anchor_logits = model(anchor_img)
            positive_features, positive_logits = model(positive_img)
            negative_features, negative_logits = model(negative_img)
            
            labels = torch.cat((anchor_label, positive_label, negative_label), dim=0)
            features = torch.cat((anchor_features, positive_features, negative_features), dim=0)
            logits = torch.cat((anchor_logits, positive_logits, negative_logits), dim=0)
            
            loss = criterion(anchor_features, positive_features, negative_features, 
                             anchor_logits, positive_logits, negative_logits,
                             anchor_label, positive_label, negative_label )
            val_loss += loss.item()
            
            euc_dist_ap = F.pairwise_distance(anchor_features, positive_features).cpu().numpy()
            euc_dist_an = F.pairwise_distance(anchor_features, negative_features).cpu().numpy()
            
            cos_dist_ap = 1 - F.cosine_similarity(anchor_features, positive_features).cpu().numpy()
            cos_dist_an = 1 - F.cosine_similarity(anchor_features, negative_features).cpu().numpy()
            
            positive_distances_euc.extend(euc_dist_ap)
            negative_distances_euc.extend(euc_dist_an)
            positive_distances_cos.extend(cos_dist_ap)
            negative_distances_cos.extend(cos_dist_an)
            
            class_preds = torch.argmax(logits, dim=1)
            all_class_preds.extend(class_preds.cpu().numpy())
            all_class_targets.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    
    classification_accuracy = np.mean(np.array(all_class_preds) == np.array(all_class_targets))
    
    # Calculate precision, recall, and F1 score for Euclidean distances
    tp_euc = np.sum(np.array(positive_distances_euc) < threshold_euc)
    fn_euc = np.sum(np.array(positive_distances_euc) > threshold_euc)
    tn_euc = np.sum(np.array(negative_distances_euc) > threshold_euc)
    fp_euc = np.sum(np.array(negative_distances_euc) < threshold_euc)

    precision_euc = tp_euc / (tp_euc + fn_euc + 1e-8)
    recall_euc = tn_euc / (tn_euc + fp_euc + 1e-8)
    score_euc = 2 * (precision_euc * recall_euc) / (precision_euc + recall_euc + 1e-8)
    # score_euc = (tp_euc + tn_euc) / (len(positive_distances_euc) + len(negative_distances_euc))

    # Calculate precision, recall, and F1 score for Cosine distances
    tp_cos = np.sum(np.array(positive_distances_cos) < threshold_cos)
    fn_cos = np.sum(np.array(positive_distances_cos) > threshold_cos)
    tn_cos = np.sum(np.array(negative_distances_cos) > threshold_cos)
    fp_cos = np.sum(np.array(negative_distances_cos) < threshold_cos)

    precision_cos = tp_cos / (tp_cos + fn_cos + 1e-8)
    recall_cos = tn_cos / (tn_cos + fp_cos + 1e-8)
    score_cos = 2 * (precision_cos * recall_cos) / (precision_cos + recall_cos + 1e-8)
    # score_cos = (tp_cos + tn_cos) / (len(positive_distances_cos) + len(negative_distances_cos))
    
    return (avg_val_loss, precision_euc, recall_euc, precision_cos, recall_cos, classification_accuracy, score_euc, score_cos)