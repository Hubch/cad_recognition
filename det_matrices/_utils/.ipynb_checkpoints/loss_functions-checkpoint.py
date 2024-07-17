import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners, distances, reducers
import sys


class AdaptiveCenterLoss(nn.Module):
    def __init__(self):
        super(AdaptiveCenterLoss, self).__init__()
        self.num_classes = None
        self.feat_dim = None
        self.centers = None

    def forward(self, features, labels):
        dtype = torch.float32 
        device = features.device
        if self.num_classes is None or self.feat_dim is None:
            self.num_classes = labels.max().item() + 1
            self.feat_dim = features.size(1)
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, device=device, dtype=dtype))

        batch_size = features.size(0)
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).t()
        distmat.addmm_(features.to(dtype), self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes, device=device, dtype=labels.dtype)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        
        return loss


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        mask = (torch.eye(2 * batch_size, device=similarity_matrix.device).bool())
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = labels.to(similarity_matrix.device)
        
        loss = self.criterion(similarity_matrix, labels)
        return loss

    
class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits, labels):
        cos_theta = logits.clamp(-1, 1)
        phi = cos_theta - self.m
        one_hot = torch.zeros(cos_theta.size(), device=logits.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        output *= self.s
        loss = F.cross_entropy(output, labels)
        return loss

    
class AngularLoss(nn.Module):
    def __init__(self, alpha=40):
        super(AngularLoss, self).__init__()
        self.alpha = alpha

    def forward(self, anchor, positive, negative):
        ap_dist = F.cosine_similarity(anchor, positive)
        an_dist = F.cosine_similarity(anchor, negative)
        loss = torch.mean(F.relu(an_dist - ap_dist + self.alpha))
        return loss
    
class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=80):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma

    def forward(self, features, labels):
        similarity_matrix = F.linear(F.normalize(features), F.normalize(features))
        one_hot = torch.zeros_like(similarity_matrix)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        pos_pair = one_hot * similarity_matrix
        neg_pair = (1 - one_hot) * similarity_matrix

        alpha_p = torch.clamp_min(-pos_pair + 1 + self.m, min=0)
        alpha_n = torch.clamp_min(neg_pair + self.m, min=0)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - self.gamma * alpha_p * (pos_pair - delta_p)
        logit_n = self.gamma * alpha_n * (neg_pair - delta_n)

        loss = F.softplus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)).mean()

        return loss
    
    
class CombinedContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.5, s=30.0, m=0.50, alpha=10, gamma=80):
        super(CombinedContrastiveLoss, self).__init__()
        self.triplet_loss = losses.TripletMarginLoss(margin=margin, distance=distances.LpDistance())
        self.triplet_cos_loss = losses.TripletMarginLoss(margin=margin, distance=distances.CosineSimilarity())
        self.nt_xent_loss = NTXentLoss(temperature)
        self.arcface_loss = ArcFaceLoss(s, m)
        self.center_loss = AdaptiveCenterLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)
        self.angular_loss = AngularLoss(alpha=alpha)
        self.circle_loss = CircleLoss(m=0.25, gamma=gamma)
        self.miner = miners.TripletMarginMiner(margin=margin, distance=distances.LpDistance(), type_of_triplets="hard")
        self.miner_cos = miners.TripletMarginMiner(margin=margin, distance=distances.CosineSimilarity(), type_of_triplets="hard")
        
        self.weights_triplet = 5.0
        self.weights_triplet_cos = 5.0
        self.weights_center = 10.0
        self.weights_ntxent = 3.0
        self.weights_arc = 3.0
        self.weights_cosine = 1.0
        self.weights_angular = 1.0
        self.weights_circle = 5.0
        
    def forward(self, anchor, positive, negative, anchor_label, positive_label, negative_label):
        embeddings = torch.cat([anchor, positive, negative], dim=0)
        labels = torch.cat([anchor_label, positive_label, negative_label], dim=0)
        
        # 硬负样本挖掘
        hard_pairs = self.miner(embeddings, labels)
        hard_pairs_cos = self.miner_cos(embeddings, labels)
        
        # Triplet Loss
        triplet_loss = self.triplet_loss(embeddings, labels, hard_pairs)
        triplet_cos_loss = self.triplet_cos_loss(embeddings, labels, hard_pairs_cos)
        
        # NT-Xent Loss
        z_i = torch.cat((anchor, positive), dim=0)
        z_j = torch.cat((positive, negative), dim=0)
        nt_xent_loss = self.nt_xent_loss(z_i, z_j)
        
        # ArcFace Loss
        arcface_loss = self.arcface_loss(embeddings, labels)
        
        # Center Loss
        center_loss = self.center_loss(embeddings, labels)
        
        # Cosine Loss
        positive_pairs = torch.cat((anchor, positive), dim=0)
        negative_pairs = torch.cat((anchor, negative), dim=0)
        positive_targets = torch.ones(positive_pairs.size(0), device=anchor.device)
        negative_targets = -torch.ones(negative_pairs.size(0), device=anchor.device)
        cosine_loss = self.cosine_loss(positive_pairs, negative_pairs, positive_targets) + \
                      self.cosine_loss(positive_pairs, negative_pairs, negative_targets)
        
        # Angular Loss
        angular_loss = self.angular_loss(anchor, positive, negative)
        
        # Circle Loss
        circle_loss = self.circle_loss(embeddings, labels)
        
        # Combine losses
        combined_loss = (self.weights_triplet * triplet_loss + \
                         self.weights_triplet_cos * triplet_cos_loss + \
                         self.weights_center * center_loss + \
                         self.weights_ntxent * nt_xent_loss + \
                         self.weights_arc * arcface_loss + \
                         self.weights_cosine * cosine_loss + \
                         self.weights_angular * angular_loss + \
                         self.weights_circle * circle_loss) / \
                        (self.weights_triplet + self.weights_triplet_cos + self.weights_center + self.weights_ntxent + self.weights_arc + self.weights_cosine + self.weights_angular + self.weights_circle)
        return combined_loss


def get_loss(margin=1.0):
    return CombinedContrastiveLoss(margin=margin,temperature=0.5, s=30.0, m=0.50, alpha=40, gamma=80)