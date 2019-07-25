import torch
from .dataset.tensor_utils import MultiLabelUnMarginLossify

def multiclass_accuracy(output, target, decimal_precision=2):
    "Computes the accuracy for the given output against target"
    with torch.no_grad():
        # get the index having the maximum output value
        _, predicted = output.max(dim=1)

        # type_as used to convert uint8 to LongTensor to avoid 
        # overflowing while calculating sum
        correct_prediction = (predicted == target).type_as(target).sum().item()

        acc = round(correct_prediction*100/target.size(0), decimal_precision)
        return acc

def multilabel_fbeta(output, target, th=0.5, alpha_for_fbeta=0, 
                     nan_to_zero=True, loss_used="BCELoss", 
                     UnMarginLossify=False):
    r"""  
    fbeta score is returned calculated according to the formula:
    f_beta = 1 / (alpha/p + (1-alpha)/r)
    where alpha is the alpha_for_fbeta provided and takes value between
    0 and 1, p is the precision and r is the recall. alpha = 1 will yield
    f_beta = precision while alpha = 0 will yield f_beta = recall.
    
    loss_used can be from the set {"BCELoss", "MultiLabelSoftMarginLoss", 
    "MultiLabelMarginLoss", "BCEWithLogitsLoss"}
    
    UnMarginLossify should be True if the target is in the form required
    by torch.nn.MultiLabelMarginLoss and needs to be converted into 
    multilabel with binary indicator.
    """
    
    assert alpha_for_fbeta >= 0 and alpha_for_fbeta <= 1, \
        'alpha should belong to the set [0,1] with inclusive boundaries.'
    assert th > 0 and th < 1, 'th should belong to the set (0,1) with '\
        'exclusive boundaries.'
    
    if loss_used in ("MultiLabelSoftMarginLoss", "BCEWithLogitsLoss"):
        output = output.sigmoid()
    
    elif loss_used == "MultiLabelMarginLoss":
        output = output.sigmoid()
        if UnMarginLossify:
            target = MultiLabelUnMarginLossify(target)
    
    elif loss_used != 'BCELoss':
        raise RuntimeError('Unsupported loss type for multilabel '
                           'classification')

    output = (output >= th)
    # elementwise multiplication of predicted and target labels
    intersec = output.float() * target
    n_intersec = torch.sum(intersec, dim=1).float()
    n_target = torch.sum(target, dim=1).float()
    n_output = torch.sum(output, dim=1).float()
    
    # mean of elementwise division for example wise p, r and f_alpha
    recall = n_intersec / n_target
    if nan_to_zero:
        recall[recall != recall] = 0
    else:
        recall = recall[recall == recall]
    recall = torch.mean(recall).item()
    
    precision = n_intersec / n_output
    if nan_to_zero:
        precision[precision != precision] = 0
    else:
        precision = precision[precision == precision]
    precision = torch.mean(precision).item()
    
    fbeta_score = 1/(alpha_for_fbeta/precision + (1-alpha_for_fbeta)/recall)
    
    return fbeta_score

class Average_Tracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count