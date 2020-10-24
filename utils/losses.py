import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV, StratifiedKFold


###############################################
# Mutual info Loss; adapted from https://github.com/fanyun-sun/InfoGraph
###############################################

def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples

    if average:
        return Eq.mean()
    else:
        return Eq


def local_global_loss_(l_enc, g_enc, batch, measure, mask):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]
    max_nodes = num_nodes // num_graphs

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    msk = torch.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    for idx, m in enumerate(mask):
        msk[idx * max_nodes + m: idx * max_nodes + max_nodes, idx] = 0.

    res = torch.mm(l_enc, g_enc.t()) * msk

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos


def global_global_loss_(g1_enc, g2_enc, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g1_enc.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).cuda()
    neg_mask = torch.ones((num_graphs, num_graphs)).cuda()
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    res = torch.mm(g1_enc, g2_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos


###############################################
# Graph ELIS Reconstruction Loss
###############################################
# def loss_rec(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
#     cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 / n_nodes * torch.mean(torch.sum(
#         1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
#     return cost + KLD

def loss_rec(preds, labels):
    pos_weight = 1 # for positive examples 
    norm = 1
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    return cost 
    # ###############################################
    # # ELIS Loss
    # ###############################################
    # def CE(self, P, Q):

    #     EPS = 1e-12

    #     losssum1 = (P * torch.log(Q + EPS)).mean()
    #     losssum2 = ((1-P) * torch.log(1-Q + EPS)).mean()
    #     losssum = -1*(losssum1 + losssum2)
        
    #     if torch.isnan(losssum):
    #         input('stop and find nan')
    #     return losssum
    # def Loss(self, latentList, input_data_index):
    #     # print('loss_ce latent weight  ',self.latentChangeForEpoch[self.epoch])
    #     loss_ce =  self.CE(P=self.P[input_data_index][:, input_data_index],
    #                         Q=self.CalPt(dist=self.Distance_squared(
    #                             latentList[0], latentList[0]),
    #                                     rho=0,
    #                                     sigma_array=1,
    #                                     gamma=self.gammaList[-1],
    #                                     v=self.vList[-1]))
    #     return loss_ce

###############################################
# Clustering KL Divergence Loss
###############################################
def loss_kldiv(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD



###############################################
# Graph Structure Loss
###############################################
def add_graph_loss(self, out_adj, features):
    # Graph regularization
    graph_loss = 0
    L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
    graph_loss += self.config['smoothness_ratio'] * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
    ones_vec = to_cuda(torch.ones(out_adj.size(-1)), self.device)
    graph_loss += -self.config['degree_ratio'] * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).squeeze() / out_adj.shape[-1]
    graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
    return graph_loss




'''
others
'''
###############################################
# Modularity Loss from https://github.com/bwilder0/clusternet/blob/master/loss_functions.py
###############################################
def loss_modularity(mu, r, embeds, dist, bin_adj, mod, args):
    bin_adj_nodiag = bin_adj*(torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
    return (1./bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()