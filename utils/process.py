import pickle as pkl
import os.path as osp
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import networkx as nx
import numpy as np
import umap

from utils.utils import Acc, visualize, compute_pairwise_similarity, cluster_acc, graph_visualize

def process(args, ):
    dist_metrics = args.dist_metrics

    if args.dataset_name in ['Cora']:
        basic_path = osp.join(osp.dirname(osp.realpath(__file__)), '..','..')
        path = osp.join(basic_path, 'data', args.dataset_name)
        # dataset = Planetoid(path, args.dataset_name, transform=T.NormalizeFeatures())
        dataset = Planetoid(path, args.dataset_name)
        
        '''
        Build clustering-oriented KNN Graph 
        '''
        print('building clustering-oriented KNN Graph ')
        x = dataset[0].x
        path = osp.join(basic_path, 'intermediate', args.dataset_name)
        if not osp.exists(path):
            os.makedirs(path)

        n_components = 2
        try: 
            feat_embedded = np.load(osp.join(path,'umap_{}.npy'.format(int(n_components))))
            print('Initial Embedding Loaded')
        except:
            reducer = umap.UMAP(n_neighbors=5, min_dist=0.7,  n_components = n_components, metric=dist_metrics)
            # feat_embedded = TSNE(n_components=2).fit_transform(x.numpy()) #.detach().cpu().numpy()
            feat_embedded = reducer.fit_transform(x.numpy())
            np.save(osp.join(path,'umap_{}.npy'.format(int(n_components))), feat_embedded)
            print('Initial Embedding Saved')
        # feat_embedded = x.numpy()
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=0, n_init=20).fit(feat_embedded)
        y_pred = kmeans.predict(feat_embedded)
        print('acc inital ', cluster_acc(dataset[0].y.numpy(), y_pred))
        
        visualize(feat_embedded, dataset[0].y.numpy())
        cluster_centers = np.zeros((args.n_clusters, feat_embedded.shape[1]))
        
        for i in range(args.n_clusters):
            cluster_centers[i, :] = np.mean(feat_embedded[y_pred == i], 0)
        real_centers_idx = pairwise_distances_argmin(feat_embedded, cluster_centers, axis=0, metric=dist_metrics) #  [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
        # real_centers_idx = dist2centers.argmin(axis=0)
        real_centers_idx = real_centers_idx.reshape(-1)
        if args.n_clusters != len(real_centers_idx):
            print('real cluster number changes')
            args.n_clusters = len(real_centers_idx)
        for i in range(args.n_clusters):
            cluster_centers[i, :] = feat_embedded[i,:]
        dist2centers = pairwise_distances(feat_embedded, cluster_centers, metric=dist_metrics) #n_sample, n_clusters
        dist2others = pairwise_distances(feat_embedded, feat_embedded, metric=dist_metrics) #n_sample, n_sample
        
        sim2clusters = np.exp(-dist2centers)/np.exp(-dist2centers).sum(1, keepdims=True) # change to t-distri
        # sim2others = np.exp(-dist2others)/np.exp(-dist2others).sum(1) 

        knn_ind_centers = np.argsort(dist2centers, axis=1) # add [::-1] if similarity measure
        knn_ind_others = np.argsort(dist2others, axis=1)

        adj_clusters = np.zeros((x.shape[0], x.shape[0]))

        np.put_along_axis(adj_clusters, knn_ind_others[:,2].reshape(-1,1), 1, axis=1) # edge sample for all
    
        for i in range(args.n_clusters): # edge sample for centers
            center_idx = real_centers_idx[i]
            temp = feat_embedded.shape[0]//args.n_clusters//10
            np.put_along_axis(adj_clusters[center_idx, :].reshape(1,-1), knn_ind_centers[center_idx, 0:temp+1].reshape(1,-1),  1, axis=1)
        adj_clusters = adj_clusters + np.transpose(adj_clusters)
        adj_clusters = np.where(adj_clusters > 0, 1, 0)
        np.fill_diagonal(adj_clusters, 0)
        for i in range(args.n_clusters-1): # add negative edges
            cidx = real_centers_idx[i]
            for j in range(i+1, args.n_clusters):
                cidx_next = real_centers_idx[j]
                adj_clusters[cidx, cidx_next] = 0
                adj_clusters[cidx_next, cidx] = 0
        
        # from sklearn.cluster import SpectralClustering

        # spCluster = SpectralClustering(n_clusters=args.n_clusters, affinity = 'precomputed', assign_labels="discretize", random_state=0).fit(adj_clusters)
        # print('acc inital for spCluster ', cluster_acc(dataset[0].y.numpy(), spCluster.labels_))

        adj_clusters_sparse = coo_matrix(adj_clusters)
        edge_index = np.concatenate((adj_clusters_sparse.row.reshape(1,-1), adj_clusters_sparse.col.reshape(1,-1)), axis=0)
        edge_weight = adj_clusters_sparse.data
        graph_visualize(feat_embedded, edge_index.transpose(), dataset[0].y.numpy(), epoch=0, acc=0)

        # os._exit(0)

        data_aux = Data(x = dataset[0].x, \
                edge_index = torch.LongTensor(edge_index),\
                edge_weight = torch.FloatTensor(edge_weight),\
                y_soft_ini = torch.FloatTensor(sim2clusters),\
                y_hard_ini = torch.FloatTensor(y_pred),\
                # cluster_centers = torch.FloatTensor(cluster_centers),
                center_indices = torch.LongTensor(real_centers_idx)) # need to improve for indices
        args.data_views['data_aux'] = data_aux
        
        
        '''
        Build main input 
        '''
        print('building main graph ')
        # import inspect

        # print(inspect.getfullargspec(to_networkx))
        G_main = to_networkx(dataset[0])
        adj_main = nx.to_numpy_array(G_main, nodelist=args.node_list)
        # G_aux = to_networkx(data_aux,remove_self_loops=True)
        # hidden = model.autoencoder.encode(data[:data_num])
        
        adj_input_main = compute_pairwise_similarity(dataset[0].x.numpy(),dataset[0].x.numpy(), dist_metrics=dist_metrics)
        
        adj_main = 0*adj_clusters + 1*adj_main
        adj_main_sparse = coo_matrix(adj_main)
        edge_index = np.concatenate((adj_main_sparse.row.reshape(1,-1), adj_main_sparse.col.reshape(1,-1)), axis=0)
        edge_weight = adj_main_sparse.data
        data_main = dataset[0]
        data_main.edge_index = torch.LongTensor(edge_index)
        data_main.edge_weight = torch.FloatTensor(edge_weight)
        

        args.data_views['data_main']  = data_main

        adjs = {'adj_input_main': adj_input_main,
                'adj_input_aux':adj_clusters}

        return args, adjs
        



# def Acc(y_pred, data):
#     # y_pred = torch.IntTensor(y_pred)
#     # print(y_pred)
#     # logits, accs = model(), []
#     # accs = []
#     correct = 0
#     total = 0

#     # print(data.y[data.train_mask])
#     # print(y_pred[data.train_mask])
#     for _, mask in data('train_mask', 'val_mask', 'test_mask'):
#         # pred = logits[mask].max(1)[1]
#         # acc = y_pred[mask].eq(data_main.y[mask]).sum().item() / mask.sum().item()
#         correct += y_pred[mask].eq(data.y[mask]).sum().item()
#         total += mask.sum().item() 
#         # print('correct', correct)
#         # print('total', total)
#         # accs.append(acc)
#     return correct/total