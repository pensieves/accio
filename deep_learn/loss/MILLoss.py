import torch
from torch import nn

EPS = 1e-7

class MILLoss(nn.Module):
    r"""Multi-Instance Learning Loss """
    
    def __init__(self, all_losses=False):
        super(MILLoss, self).__init__()
        self.all_losses = all_losses
    
    def forward(self, probas, labels, bag_ids, neg_weight=1, pos_weight=1):

        # calculate negative region loss
        neg_idx_select = labels == 0
        num_neg_ids = bag_ids[neg_idx_select].unique().size(0)
        
        weighted_neg_loss = 0
        
        if num_neg_ids > 0:
            neg_bag_probas = probas[neg_idx_select]
            weighted_neg_loss = -(neg_weight * torch.log(1 - neg_bag_probas + 
                                                     EPS)).sum() / num_neg_ids
        
        # calculate positive region loss
        pos_idx_select = labels == 1

        pos_ids = bag_ids[pos_idx_select]
        unique_pos_ids = pos_ids.unique()
        num_pos_ids = unique_pos_ids.size(0)

        weighted_pos_loss = 0
        
        if num_pos_ids > 0:
            pos_bag_probas = probas[pos_idx_select]

            pos_bag_log_not_probas = torch.log(1 - pos_bag_probas + EPS)

            pos_ids_presence_expanded_over_dim = \
                            (pos_ids == unique_pos_ids.view(-1,1))\
                                .type(torch.FloatTensor).to(probas.device)

            # clamping max proba value to 1 to subdue proba overshoot because 
            # of EPS used at the time of log computation above.
            all_neg_proba_pos_bag_ids = torch.exp((pos_bag_log_not_probas*
                pos_ids_presence_expanded_over_dim).sum(dim=1)).clamp(max=1)

            weighted_pos_loss = -(pos_weight * 
                                  torch.log(1 - all_neg_proba_pos_bag_ids + 
                                            EPS)).sum() / num_pos_ids

        # aggregate losses
        weighted_total_loss = (weighted_neg_loss + weighted_pos_loss)
        
        if self.all_losses is False:
            return weighted_total_loss
        return weighted_total_loss, weighted_neg_loss, weighted_pos_loss
