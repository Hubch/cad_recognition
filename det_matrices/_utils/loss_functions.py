import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners, distances
import json
from copy import deepcopy

# BYOL the more effecient self-surprised loss
class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=4096):
        super(MLPHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class BYOL(nn.Module):
    def __init__(self, base_encoder, projection_dim=128, moving_average_decay=0.99):
        super(BYOL, self).__init__()
        self.online_encoder = base_encoder
        self.target_encoder = deepcopy(base_encoder)

        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        self.feature_dim = base_encoder.fc_feature.out_features
        self.online_projector = MLPHead(self.feature_dim, projection_dim)
        self.target_projector = MLPHead(self.feature_dim, projection_dim)
        self.online_predictor = MLPHead(projection_dim, projection_dim)
        self.moving_average_decay = moving_average_decay

    @torch.no_grad()
    def update_target_encoder(self):
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.moving_average_decay * target_params.data + (1 - self.moving_average_decay) * online_params.data

    def forward(self, x1, x2):
        online_feature1, _ = self.online_encoder(x1)
        online_feature2, _ = self.online_encoder(x2)
        
        online_proj1 = self.online_projector(online_feature1)
        online_proj2 = self.online_projector(online_feature2)
        online_pred1 = self.online_predictor(online_proj1)
        online_pred2 = self.online_predictor(online_proj2)

        with torch.no_grad():
            target_feature1, _ = self.target_encoder(x1)
            target_feature2, _ = self.target_encoder(x2)
            target_proj1 = self.target_projector(target_feature1)
            target_proj2 = self.target_projector(target_feature2)

        loss1 = F.mse_loss(online_pred1, target_proj2.detach())
        loss2 = F.mse_loss(online_pred2, target_proj1.detach())
        return 0.5 * (loss1 + loss2)
    
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

class CostSensitiveLoss(nn.Module):
    def __init__(self, base_loss_fn, class_weights):
        super(CostSensitiveLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def forward(self, logits, labels):
        device = logits.device
        self.class_weights = self.class_weights.to(device)
        weights = self.class_weights[labels]
        loss = self.base_loss_fn(logits, labels)
        return (loss * weights).mean()

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, teacher_logits, student_logits, teacher_features, student_features):
        # Classification Distillation Loss
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        classification_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
        
        # Feature Alignment Loss
        feature_alignment_loss = F.mse_loss(student_features, teacher_features)

        # Combine the losses
        loss = 0.5 * (classification_loss + feature_alignment_loss)

        return loss
    
    
class CombinedLoss(nn.Module):
    def __init__(self, model, margin=1.0, num_classes=1000, num_epochs=0, temperature=4.0, s=30.0, m=0.50, weight_decay=1e-3, smoothing_factor=0.9, initial_weights=None):
        super(CombinedLoss, self).__init__()

        # Load the class weight
        with open('./weights/class_weights.json', 'r') as f:
            class_weights = json.load(f)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        
        self.initialized = False
        self.model = model
        self.triplet_cos_loss = losses.TripletMarginLoss(margin=margin, distance=distances.CosineSimilarity())
        self.nt_xent_loss = NTXentLoss(temperature)
        self.arcface_loss = ArcFaceLoss(s, m)
        self.classification_loss = CostSensitiveLoss(nn.CrossEntropyLoss(reduction='none'), class_weights)
        self.distillation_loss = DistillationLoss(temperature=temperature)
        self.miner_cos = miners.TripletMarginMiner(margin=margin, distance=distances.CosineSimilarity(), type_of_triplets="hard")
        
        self.weights_triplet_cos = 3.0
        self.weights_ntxent = 2.0
        self.weights_arc = 1.0
        self.weights_classification = 3.0
        self.weights_self_surprised = 1.0
        self.weights_distillation = 1.0
        self.weight_decay = weight_decay
        
        # Initialize EMA for weights
        self.ema_weights = None
        self.ema_alpha = 0.1  # Smoothing factor for EMA
        self.upper_bound = 0.5
        self.lower_bound = 0.1
        
    def get_gradient_norm_loss_weights(self, losses):
        grads = []
        valid_losses = []
        
        for loss in losses:
            if loss.requires_grad:
                grad = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True, allow_unused=True)
                grads.append(grad)
                valid_losses.append(loss)

        norms = []
        for grad in grads:
            norm = sum(torch.norm(g).item() for g in grad if g is not None)  # Avoid None gradients
            norms.append(norm)

        total_norm = sum(norms)
        weights = [total_norm / (norm if norm > 0 else 1) for norm in norms]  # Avoid division by zero
        
        # Clip weights to avoid any single loss from dominating
        weights = [max(self.lower_bound, min(w, self.upper_bound)) for w in weights]
        
        # Normalize weights
        normalized_weights = [w / sum(weights) for w in weights]

        # Apply EMA smoothing
        if self.ema_weights is None:
            self.ema_weights = normalized_weights
        else:
            self.ema_weights = [self.ema_alpha * w + (1 - self.ema_alpha) * ema_w for w, ema_w in zip(normalized_weights, self.ema_weights)]

        return self.ema_weights, valid_losses
    
    def l2_regularization(self):
        l2_reg = torch.tensor(0.).to(next(self.parameters()).device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param, 2)
        return l2_reg

    def forward(self, anchor_features, positive_features, negative_features, 
                anchor_logits, positive_logits, negative_logits, 
                anchor_labels, positive_labels, negative_labels, 
                byol_loss=None, teacher_logits=None, teacher_features=None,):
        
        embeddings = torch.cat([anchor_features, positive_features, negative_features], dim=0)
        logits = torch.cat([anchor_logits, positive_logits, negative_logits], dim=0)
        labels = torch.cat([anchor_labels, positive_labels, negative_labels], dim=0)
        
        # 负样本挖掘
        hard_pairs_cos = self.miner_cos(embeddings, labels)
        
        # Triplet Loss
        triplet_cos_loss = self.triplet_cos_loss(embeddings, labels, hard_pairs_cos)
        
        # NT-Xent Loss
        z_i = torch.cat((anchor_features, positive_features), dim=0)
        z_j = torch.cat((positive_features, negative_features), dim=0)
        nt_xent_loss = self.nt_xent_loss(z_i, z_j)
        
        # ArcFace Loss
        arcface_loss = self.arcface_loss(embeddings, labels)
        
        # Classification Loss
        classification_loss = self.classification_loss(logits, labels)

        # BYOL Loss
        if byol_loss is None:
            byol_loss = torch.tensor(0.0, device=embeddings.device)

        # Distillation Loss
        if teacher_logits is None:
            distillation_loss = torch.tensor(0.0, device=embeddings.device)
        else:
            distillation_loss = self.distillation_loss(teacher_logits, anchor_logits, teacher_features, anchor_features)
        
        combined_loss = self.weight_decay * self.l2_regularization() + (
            self.weights_triplet_cos * triplet_cos_loss +
            self.weights_ntxent * nt_xent_loss +
            self.weights_arc * arcface_loss +
            self.weights_classification * classification_loss +
            self.weights_self_surprised * byol_loss +
            self.weights_distillation * distillation_loss
        ) / (
            self.weights_triplet_cos + self.weights_ntxent + self.weights_arc + 
            self.weights_classification + self.weights_self_surprised + self.weights_distillation
        )
        
        return combined_loss
        
#         # L2 Regularization
#         l2_reg = torch.tensor(0.).to(embeddings.device)
#         for param in self.parameters():
#             l2_reg += torch.norm(param, 2)
            
#         loss_components = [triplet_cos_loss, nt_xent_loss, arcface_loss, classification_loss, byol_loss, distillation_loss]
        
#         dynamic_weights, valid_losses = self.get_gradient_norm_loss_weights(loss_components)
#         combined_loss = self.weight_decay * l2_reg + sum(w * loss for w, loss in zip(dynamic_weights, valid_losses))
         
#         return combined_loss

def get_loss(model=None, margin=1.0, num_classes=None, num_epochs=0):
    return CombinedLoss(model=model, margin=margin, num_classes=num_classes, num_epochs=num_epochs, temperature=4.0, s=30.0, m=0.50)