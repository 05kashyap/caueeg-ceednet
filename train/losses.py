import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        class_weights: Optional class weights tensor
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, class_weights=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss with automatic class weight handling.
    """
    
    def __init__(self, class_weights=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.class_weights, reduction=self.reduction)


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for multi-label classification.
    """
    
    def __init__(self, class_weights=None, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # Apply class weights if provided
        if self.class_weights is not None:
            # Expand weights to match target shape
            weights = self.class_weights.unsqueeze(0).expand_as(targets)
            loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weights, reduction='none')
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            return F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)


def get_loss_function(criterion_name, class_weights=None, **kwargs):
    """
    Factory function to get the appropriate loss function.
    
    Args:
        criterion_name: Name of the loss function
        class_weights: Optional class weights tensor
        **kwargs: Additional arguments for loss functions
    
    Returns:
        Loss function instance
    """
    
    if criterion_name == "cross-entropy":
        return WeightedCrossEntropyLoss(class_weights=class_weights)
    
    elif criterion_name == "weighted-cross-entropy":
        return WeightedCrossEntropyLoss(class_weights=class_weights)
    
    elif criterion_name == "focal":
        alpha = kwargs.get('focal_alpha', 1.0)
        gamma = kwargs.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma, class_weights=class_weights)
    
    elif criterion_name == "multi-bce":
        return WeightedBCELoss(class_weights=class_weights)
    
    elif criterion_name == "weighted-multi-bce":
        return WeightedBCELoss(class_weights=class_weights)
    
    elif criterion_name == "svm":
        # Note: Multi-margin loss doesn't directly support class weights
        # You might need to implement a custom weighted version
        return nn.MultiMarginLoss()
    
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")


def compute_loss_with_criterion(output, targets, criterion_name, class_weights=None, **kwargs):
    """
    Compute loss using the specified criterion with proper handling of different loss types.
    
    Args:
        output: Model output logits
        targets: Target labels
        criterion_name: Name of the loss criterion
        class_weights: Optional class weights
        **kwargs: Additional arguments for loss functions
    
    Returns:
        loss: Computed loss
        predictions: Predictions for accuracy calculation
    """
    
    if criterion_name == "cross-entropy" or criterion_name == "weighted-cross-entropy":
        if class_weights is not None:
            loss = F.cross_entropy(output, targets, weight=class_weights)
        else:
            loss = F.cross_entropy(output, targets)
        predictions = F.log_softmax(output, dim=1)
    
    elif criterion_name == "focal":
        alpha = kwargs.get('focal_alpha', 1.0)
        gamma = kwargs.get('focal_gamma', 2.0)
        focal_loss = FocalLoss(alpha=alpha, gamma=gamma, class_weights=class_weights)
        loss = focal_loss(output, targets)
        predictions = F.log_softmax(output, dim=1)
    
    elif criterion_name == "multi-bce" or criterion_name == "weighted-multi-bce":
        targets_oh = F.one_hot(targets, num_classes=output.size(1)).float()
        if class_weights is not None and criterion_name == "weighted-multi-bce":
            weights = class_weights.unsqueeze(0).expand_as(targets_oh)
            loss = F.binary_cross_entropy_with_logits(output, targets_oh, weight=weights)
        else:
            loss = F.binary_cross_entropy_with_logits(output, targets_oh)
        predictions = torch.sigmoid(output)
    
    elif criterion_name == "svm":
        loss = F.multi_margin_loss(output, targets)
        predictions = output
    
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")
    
    return loss, predictions
