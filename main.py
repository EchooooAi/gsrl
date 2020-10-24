import os.path as osp
import os
import argparse

from utils.process import process
from utils.utils import Acc, visualize, cluster_acc
import torch
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--device',  type=str, default= 'cpu',
                    help='device')
#data
parser.add_argument('--dataset_name', type=str, default='Cora',
                    help='name of used dataset')
parser.add_argument('--data_views', type=dict, default={'data_main':None, 'data_aux':None,},
                    help='storage for PyG Data class')
parser.add_argument('--node_list',  type=list, default= None,
                    help='node_list_order')
#model
parser.add_argument('--n_clusters',  type=int, default= 7,
                    help='number of clusters')
parser.add_argument('--dist_metrics',  type=str, default= 'euclidean',
                    help='distance metrics: support euclidean')
parser.add_argument('--dropout',  type=float, default= 0.5,
                    help='dropout rate')


args = parser.parse_args()
# if args.use_gdc:
#     gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
#                 normalization_out='col',
#                 diffusion_kwargs=dict(method='ppr', alpha=0.05),
#                 sparsification_kwargs=dict(method='topk', k=128,
#                                            dim=0), exact=True)
#     data = gdc(data)


args, adjs = process(args)




from models.graphsrl import GraphSRL




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device
model = GraphSRL(args).to(device)
# model, args, adjs = GraphSRL(args).to(device), args.to(device), adjs.to(device)
lr = 0.01
print('dict(params=model.mlp_node_rep.parameters()', dict(params=model.mlp_node_rep.parameters()))
optimizer = torch.optim.Adam([
    dict(params=model.gcn_encoder_main.parameters(), weight_decay=0, lr =0.1),
    dict(params=model.mlp_node_rep.parameters(), weight_decay=0),
    dict(params=model.mlp_read_out.parameters(), weight_decay=0)
], lr=lr)  
optimizer_aux = torch.optim.Adam([
    dict(params=model.gcn_encoder_aux.parameters(), weight_decay=0),
    dict(params=model.mlp_node_rep.parameters(), weight_decay=0),
    dict(params=model.mlp_read_out.parameters(), weight_decay=0)
], lr=lr)  

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer_aux = torch.optim.Adam(model.parameters(), lr=lr)


data_main = args.data_views["data_main"].to(device)
data_aux = args.data_views["data_aux"].to(device)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def train():
    model.train()
    optimizer.zero_grad()
    # F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    output = model(data_main, data_aux)
    losses = model.comput_loss(data_main, data_aux, output, adjs)
    optimizer.zero_grad()
    set_requires_grad(model.gcn_encoder_aux, requires_grad=False)
    losses['loss_main'].backward(retain_graph=True)
    optimizer.step()

    set_requires_grad(model.gcn_encoder_aux, requires_grad=True)
    set_requires_grad(model.gcn_encoder_main, requires_grad=False)
    optimizer_aux.zero_grad()
    losses['loss_aux'].backward()
    optimizer_aux.step()
    set_requires_grad(model.gcn_encoder_main, requires_grad=True)
    return losses, output


@torch.no_grad()
def test():
    model.eval()
    output = model(data_main, data_aux)
    latent = output['latent_main'].detach().cpu().numpy()
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0, n_init=20).fit(latent)
    # y_pred = torch.IntTensor(kmeans.predict(latent)).to(device)
    y_pred = kmeans.predict(latent)
    acc = cluster_acc(data_main.y.detach().cpu().numpy(), y_pred)
    return acc


# best_val_acc = test_acc = 0
loss_history = [0,0]
for epoch in range(1, 201):
    losses, output = train()
    loss_history[0] +=  losses['loss_main'].item()
    loss_history[1] +=  losses['loss_aux'].item()
    # train_acc, val_acc, tmp_test_acc = test()
    acc = test()
    if epoch%20 == 0:
        visualize(output['latent_main'].detach().cpu().numpy(), data_main.y.detach().cpu().numpy(), epoch)
    # log = 'Epoch: {:03d}, loss main:{:.4f}, loss aux:{:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    log = 'Epoch: {:03d}, loss main:{:.4f}, loss aux:{:.4f}, Acc: {:.4f},'
    print(log.format(epoch, loss_history[0]/epoch, loss_history[1]/epoch, acc))