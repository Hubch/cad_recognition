from utils.models import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_distances(device, model, data_loader):
    model.eval()
    positive_distances_euclid = []
    negative_distances_euclid = []
    positive_distances_cosine = []
    negative_distances_cosine = []

    with torch.no_grad():
        for anchor_img, positive_img, negative_img, anchor_label, positive_label, negative_label in data_loader:
            anchor_img, positive_img, negative_img = anchor_img.to(device), positive_img.to(device), negative_img.to(device)

            anchor_feature = model(anchor_img)
            positive_feature = model(positive_img)
            negative_feature = model(negative_img)

            positive_distance_euclid = F.pairwise_distance(anchor_feature, positive_feature).cpu().numpy()
            negative_distance_euclid = F.pairwise_distance(anchor_feature, negative_feature).cpu().numpy()
            positive_distance_cosine = 1 - F.cosine_similarity(anchor_feature, positive_feature).cpu().numpy()
            negative_distance_cosine = 1 - F.cosine_similarity(anchor_feature, negative_feature).cpu().numpy()
            positive_distances_euclid.extend(positive_distance_euclid)
            negative_distances_euclid.extend(negative_distance_euclid)
            positive_distances_cosine.extend(positive_distance_cosine)
            negative_distances_cosine.extend(negative_distance_cosine)

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


    # 使用均值和标准差来确定阈值范围
    lower_bound_euc = positive_mean_euc + positive_std_euc
    upper_bound_euc = negative_mean_euc - negative_std_euc

    lower_bound_cos = positive_mean_cos + positive_std_cos
    upper_bound_cos = negative_mean_cos - negative_std_cos

    # 设置步长和范围
    steps = 1000
    step_size_euc = (upper_bound_euc - lower_bound_euc) / steps
    step_size_cos = (upper_bound_cos - lower_bound_cos) / steps

    best_threshold_euc = lower_bound_euc
    best_threshold_cos = lower_bound_cos
    best_score_euc = 0
    best_score_cos = 0

    for i in range(steps):
        threshold_euc = lower_bound_euc + i * step_size_euc
        correct_positive_euc = np.sum(positive_distances_euc < threshold_euc)
        correct_negative_euc = np.sum(negative_distances_euc >= threshold_euc)
        score_euc = correct_positive_euc + correct_negative_euc

        threshold_cos = lower_bound_cos + i * step_size_cos
        correct_positive_cos = np.sum(positive_distances_cos < threshold_cos)
        correct_negative_cos = np.sum(negative_distances_cos >= threshold_cos)
        score_cos = correct_positive_cos + correct_negative_cos

        if score_euc > best_score_euc:
            best_score_euc = score_euc
            best_threshold_euc = threshold_euc

        if score_cos > best_score_cos:
            best_score_cos = score_cos
            best_threshold_cos = threshold_cos

    return best_threshold_euc, best_threshold_cos

def evaluate_model(device, model, criterion, data_loader, threshold_euc, threshold_cos):
    model.eval()
    same_class_correct_euc = 0
    different_class_correct_euc = 0
    same_class_total_euc = 0
    different_class_total_euc = 0

    same_class_correct_cos = 0
    different_class_correct_cos = 0
    same_class_total_cos = 0
    different_class_total_cos = 0

    epoch_loss = .0

    with torch.no_grad():
        for anchor, positive, negative, anchor_label, positive_label, negative_label in data_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_label, positive_label, negative_label = anchor_label.to(device), positive_label.to(device), negative_label.to(device)

            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            positive_distance_euc = F.pairwise_distance(anchor_output, positive_output).cpu().numpy()
            negative_distance_euc = F.pairwise_distance(anchor_output, negative_output).cpu().numpy()
            positive_distance_cos = 1 - F.cosine_similarity(anchor_output, positive_output).cpu().numpy()
            negative_distance_cos = 1 - F.cosine_similarity(anchor_output, negative_output).cpu().numpy()

            same_class_total_euc += len(positive_distance_euc)
            different_class_total_euc += len(negative_distance_euc)
            same_class_correct_euc += np.sum(positive_distance_euc < threshold_euc)
            different_class_correct_euc += np.sum(negative_distance_euc >= threshold_euc)

            same_class_total_cos += len(positive_distance_cos)
            different_class_total_cos += len(negative_distance_cos)
            same_class_correct_cos += np.sum(positive_distance_cos < threshold_cos)
            different_class_correct_cos += np.sum(negative_distance_cos >= threshold_cos)

            loss = criterion(anchor_output, positive_output, negative_output, anchor_label, positive_label, negative_label)
            epoch_loss += loss.item()

    same_class_accuracy_euc = same_class_correct_euc / same_class_total_euc if same_class_total_euc else 0
    different_class_accuracy_euc = different_class_correct_euc / different_class_total_euc if different_class_total_euc else 0
    same_class_accuracy_cos = same_class_correct_cos / same_class_total_cos if same_class_total_cos else 0
    different_class_accuracy_cos = different_class_correct_cos / different_class_total_cos if different_class_total_cos else 0

    avg_epoch_loss = epoch_loss / len(data_loader)
    print(f'Val Loss: {avg_epoch_loss}')

    return avg_epoch_loss, same_class_accuracy_euc, different_class_accuracy_euc, same_class_accuracy_cos, different_class_accuracy_cos