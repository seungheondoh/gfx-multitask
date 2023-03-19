import torch
import torch.nn as nn

class ContrastiveHead(nn.Module):
    def __init__(self, temperature=0.2):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        """
        pos: batch x 1 (sim score)
        neg: batch x neg (sim score)
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        loss = self.criterion(logits, labels)
        return loss
    
    def precision_at_k(output, target, top_k=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

class ClsHead(nn.Module):
    """Simplest classifier head, with only one fc layer.
    """
    def __init__(self, in_channels, probe_type, num_classes, with_avg_pool=True):
        super(ClsHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.probe_type = probe_type
        self.criterion = nn.CrossEntropyLoss()
        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if (self.probe_type == "linear") or (self.probe_type == "ft"):
            self.fc_cls = nn.Sequential(
                    nn.LayerNorm(in_channels), nn.Linear(in_channels, num_classes)
            )
        elif self.probe_type == "mlp":
            self.fc_cls = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, num_classes),
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
        output = self.fc_cls(x)
        logits = self.sigmoid(output)
        return logits