import torch
import torch.nn as nn


class BertEDL(nn.Module):
    def __init__(self, num_classes=2, hidden_dims=[512, 256], dropout=0.3, input_dim=768):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim  # 预提取 BERT 特征维度
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        # 最后一层线性层输出 logits
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, features):
        # 多层分类器
        logits = self.classifier(features)

        # EDL 核心：证据必须非负
        evidence = torch.relu(logits)
        alpha = evidence + 1.0
        return alpha


def edl_loss(alpha, labels, num_classes=2, annealing=0.01):
    S = torch.sum(alpha, dim=1, keepdim=True)
    label_onehot = torch.nn.functional.one_hot(labels, num_classes).float()

    # 1. 核心损失
    loss = torch.sum(label_onehot * (torch.log(S) - torch.log(alpha)), dim=1)

    # 2. KL 正则（论文原版，最简单最稳）
    beta = torch.ones_like(alpha)
    S_b = beta.sum(1, keepdim=True)
    kl = torch.lgamma(S).squeeze() - torch.lgamma(alpha).sum(1) \
         - torch.lgamma(S_b).squeeze() + torch.lgamma(beta).sum(1) \
         + torch.sum((beta - 1) * (torch.digamma(alpha) - torch.digamma(S)), dim=1)

    # 3. 总和
    total_loss = loss + annealing * kl
    return total_loss.mean()