import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

class MLP(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.ReLU(),
            # nn.Linear(out_ft, out_ft),
            # nn.ReLU(),
            # nn.Linear(out_ft, out_ft),
            # nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        # return self.ffn(x) + self.linear_shortcut(x)
        return self.ffn(x)
        
class GraphSRL(torch.nn.Module):
    def __init__(self, args, dims_latent=16):
        super(GraphSRL, self).__init__()
        self.data_main = args.data_views['data_main']
        self.data_aux = args.data_views['data_aux']
        self.center_indices = self.data_aux.center_indices
        self.num_features = self.data_main.x.size(1)
        self.num_classes = self.data_main.y.max()+1
        self.num_nodes = self.data_main.x.size(0)
        self.dist_metrics = args.dist_metrics
        self.device = args.device

        # self.gcn_encoder_main = nn.Sequential(
        #     GCNConv(self.num_features, dims_latent),
        #     nn.PReLU(),
        #     nn.Dropout(p=args.dropout),
        #     GCNConv(dims_latent, dims_latent),
        #     nn.PReLU(),
        # )
        # self.gcn_encoder_aux = nn.Sequential(
        #     GCNConv(self.num_features, dims_latent),
        #     nn.PReLU(),
        #     nn.Dropout(p=args.dropout),
        #     GCNConv(dims_latent, dims_latent),
        #     nn.PReLU(),
        # )
        dims_temp = 128
        self.gcn_encoder_main = nn.ModuleList([
            GCNConv(self.num_features, dims_temp),
            GCNConv(dims_temp, dims_latent),
            ])
        self.gcn_encoder_aux = nn.ModuleList([
            GCNConv(self.num_features, dims_temp),
            GCNConv(dims_temp, dims_latent),
        ])
        self.act = nn.PReLU()
        self.dropout = nn.Dropout(p=args.dropout)


        self.mlp_node_rep = MLP(dims_latent, dims_latent)
        self.mlp_read_out = MLP(self.num_nodes * dims_latent, dims_latent)

        # self.conv1 = GCNConv(dataset.num_features, 16)
        # self.conv2 = GCNConv(16, dataset.num_classes)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)
    def compute_pairwise_similarity(self, x, y, std=False, to_distribution=False):
        if self.dist_metrics == 'euclidean':
            freedom = 1 
            squared = torch.sum( torch.pow((x.unsqueeze(1) - y),2), 2) # ret n_sample, n_cluster 
            sim_ij = 1.0 / (1.0 + (squared / freedom))
            sim_ij = torch.pow(sim_ij, float(freedom + 1)/2)
       
        elif self.dist_metrics == 'cosine':
            sim_ij = x.mm(y.transpose(0,1))
            sim_ij = torch.sigmoid(sim_ij)
            # return F.cosine_similarity(x, y.unsqueeze(1), dim=-1)
        if std:
            means = sim_ij.mean(dim=1, keepdim=True)
            stds = sim_ij.std(dim=1, keepdim=True)
            sim_ij = (sim_ij - means) / stds
        if to_distribution:
            sim_ij = (sim_ij.t() / torch.sum(sim_ij, 1)).t() # (n_samples, n_clusters)
        return sim_ij



    def forward(self, data_main, data_aux):
        # data_main = args.dataset.data
        # data_aux = args.dataset.data_aux
        x, edge_index, edge_weight = data_main.x, data_main.edge_index, data_main.edge_weight
        x2, edge_index2, edge_weight2 = data_aux.x, data_aux.edge_index, data_aux.edge_weight
        
        # x = self.gcn_encoder_main(x, edge_index, edge_weight)
        # x2 = self.gcn_encoder_aux(x2, edge_index2, edge_weight2)
        for encoder in self.gcn_encoder_main:
            x = encoder(x, edge_index, edge_weight)
            self.act(x)
            # self.dropout(x)

        for encoder in self.gcn_encoder_aux:
            x2 = encoder(x2, edge_index2, edge_weight2)
            self.act(x)
            # self.dropout(x)
        # x = self.gcn_encoder_main(x, edge_index)
        # x2 = self.gcn_encoder_aux(x2, edge_index2)
        
        # x_latent = self.mlp_node_rep(x)
        # x2_latent = self.mlp_node_rep(x2)
        x_latent = x
        x2_latent = x2

        x_summary = self.mlp_read_out(x_latent.view(-1))
        x2_summary = self.mlp_read_out(x2_latent.view(-1))


        x_decode = self.compute_pairwise_similarity(x_latent, x_latent, std=False)
        x2_decode = self.compute_pairwise_similarity(x2_latent, x2_latent, std=False)
        # x_decode = torch.sigmoid(x_decode) #sigmoid
        # x2_decode = torch.sigmoid(x2_decode)

        # x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index, edge_weight)
        output = {'latent_main': x_latent, 'latent_aux': x2_latent, 
                'summary_main': x_summary, 'summary_aux': x2_summary, 
                'decode_main': x_decode, 'decode_aux': x2_decode}
        return output


    def loss_rec(self, adj_rec, adj_input, weight = 1):
        '''
        Reconstruction Loss
        '''
        # pos_weight = 1 # for positive examples 
        #pos_weight=pos_weight
        loss = weight * F.binary_cross_entropy(adj_rec, adj_input)
       
        # EPS = 1e-12
        # losssum1 = (adj_input * torch.log(adj_rec + EPS)).mean()
        # losssum2 = ((1 - adj_input) * torch.log(1- adj_rec + EPS)).mean()
        # losssum = -1*(losssum1 + losssum2)
        # if torch.isnan(losssum):
        #     input('stop and find nan')

        return loss
    
    def loss_kldiv(self, q_latent, p_latent, weight = 1):

        weight = 1
        # if self.center_indices == None:
        #     print('Error: No cluster centers input')
        #     return 0
    
        q_ij = self.compute_pairwise_similarity(q_latent, q_latent[self.center_indices,:], to_distribution=True)
        p_ij = self.compute_pairwise_similarity(p_latent, p_latent[self.center_indices,:], to_distribution=True)
        
        loss = weight * F.kl_div(q_ij.log(), p_ij, reduction='batchmean') # KL(p = target || q = input)
        return loss

    def loss_structure(self, x ,adj_rec, weight = 1):
        '''
        Reconstruction Loss
        '''
        loss = 0
        smoothness_ratio = 0
        degree_ratio = 0
        sparsity_ratio = 0
        ones_vec = torch.ones(adj_rec.size(-1)).to(self.device)
        L = torch.diagflat(torch.sum(adj_rec, -1)) - adj_rec
        # loss += smoothness_ratio * torch.trace(torch.mm(self.data_main.x.transpose() torch.mm(L, data_main.x))) / int(np.prod(adj_rec.shape))
        loss += smoothness_ratio * torch.trace(torch.mm( x.transpose(0,1), torch.mm(L, x) ))/x.size(0)
        loss -= degree_ratio * \
            torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(adj_rec, ones_vec.unsqueeze(-1)) + 1e-15 )).squeeze() / adj_rec.shape[-1]

        loss += sparsity_ratio * torch.sum(torch.pow(adj_rec, 2))
  

        return loss*weight


    def comput_loss(self, data_main, data_aux, output, adjs):

        '''
        adj_input_main :torch tensor
        adj_input_aux :torch tensor
        '''

        y_main = data_main.y
        y_aux = data_aux.y

        adj_input_main = torch.FloatTensor(adjs['adj_input_main']).to(self.device)
        adj_input_aux = torch.FloatTensor(adjs['adj_input_aux']).to(self.device)

        loss1 = 0 # main
        loss2 = 0
        loss1 += self.loss_rec(output['decode_main'], adj_input_main, weight=1)
        loss1 += self.loss_kldiv(output['latent_main'], output['latent_aux'].detach(), weight=1)
        loss1 += self.loss_structure(data_main.x, output['decode_main'], weight=0)
        loss2 += self.loss_rec(output['decode_aux'], adj_input_aux, weight=1)

        losses = {'loss_main':loss1, 'loss_aux':loss2}
        return losses