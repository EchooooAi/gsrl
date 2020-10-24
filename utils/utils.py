import torch
from matplotlib import pyplot as plt
import os
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
import numpy as np
import networkx as nx

import warnings 
import matplotlib.cbook 
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


basic_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..','..')

def visualize(feat, labels, epoch=0, y_pred=None, data=None, args=None):
    path = os.path.join(basic_path, 'res', 'visualization')
    if not os.path.exists(path):
        os.makedirs(path)
	# hidden = model.autoencoder.encode(data).detach().cpu().numpy()	
	# np.save("{}data/hidden_{}.npy".format(path, epoch), hidden)
	# np.save("{}data/centers_{}.npy".format(path, epoch), model.cluster_centers.detach().cpu().numpy())
	# np.save("{}data/y_pred_{}.npy".format(path, epoch), y_pred)
	# reducer = umap.UMAP(n_neighbors=5, min_dist=0.7,  metric='correlation')
	# x_embedded = reducer.fit_transform(hidden)
	
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111)
    plt.scatter(feat[:,0], feat[:,1], c=labels, s=5, cmap='rainbow_r')

	# output = model(data)[1].argmax(1).detach().cpu().numpy()
    if y_pred is not None and data is not None:
        acc = Acc(y_pred, data)
        # nmi = nmi_score(labels, output)
    else:
        acc = 0
    fig.savefig( os.path.join(path, 'cora_{}_{}.png'.format(round(acc, 4), epoch)))
    print('figure save in ', path)
    plt.close(fig)

	# if epoch == 0:
	# 	fig = plt.figure()
	# 	ax = plt.subplot(111)
	# 	plt.scatter(x_embedded[:len(y_pred), 0], x_embedded[:len(y_pred), 1], c=y_pred, s=1, cmap='rainbow_r')
	# 	fig.savefig('{}pics/Cluster_mnist.png'.format(path))
	# 	plt.close(fig)

	# 	np.save("{}data/input.npy".format(path), data.cpu().numpy())
	# 	np.save("{}data/labels.npy".format(path), labels)


def graph_visualize(feat, edge_list, labels, epoch=0, acc=0):
    '''
    edge_list: n * 2 numpy array
    '''
    G = nx.Graph()
    G.add_nodes_from(list(range(feat.shape[0])))
    G.add_edges_from(list(map(tuple, edge_list))) 

    # zip(list(range(labels.max()+1)), list(range(labels.max()+1)))
    # val_map = {0: 1.0,
    #         # 'D': 0.5714285714285714,
    #         4: 1}
    # values = [val_map.get(node, 0.5) for node in G.nodes()]
    values = list(labels)
    fig = plt.figure(figsize=(8, 8))
    for node_idx in range(feat.shape[0]):
        G.node[node_idx]['pos'] = feat[node_idx]
    # pos = nx.spring_layout(G)
    pos=nx.get_node_attributes(G,'pos')

    nx.draw(G, pos=pos, alpha=1 ,cmap=plt.get_cmap('rainbow'), node_color=values, style='solid',node_size=10)
    
    path = os.path.join(basic_path, 'res', 'graphVis')
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig( os.path.join(path, 'cora_{}_{}.png'.format(round(acc, 4), epoch)))
    print('figure save in ', path)
    plt.close(fig)


    # nx.draw_networkx_edges(
    #     G,
    #     pos,
    #     edgelist=[(0,4)],
    #     width=1,
    #     alpha=1,
    #     edge_color="b",
    # )


def Acc(y_pred, data):
    # y_pred = torch.IntTensor(y_pred)
    # print(y_pred)
    # logits, accs = model(), []
    # accs = []
    correct = 0
    total = 0

    # print(data.y[data.train_mask])
    # print(y_pred[data.train_mask])
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        # pred = logits[mask].max(1)[1]
        # acc = y_pred[mask].eq(data_main.y[mask]).sum().item() / mask.sum().item()
        correct += y_pred[mask].eq(data.y[mask]).sum().item()
        total += mask.sum().item() 
        # print('correct', correct)
        # print('total', total)
        # accs.append(acc)
    return correct/total

def compute_pairwise_similarity(x, y, dist_metrics='euclidean', std=False, to_distribution=False):
    if dist_metrics == 'euclidean':
        sim_ij = pairwise_distances(x, y, metric=dist_metrics) #n_sample, n_sample
        freedom = 1 
        sim_ij = 1.0 / (1.0 + (np.power(sim_ij, 2) / freedom))
        sim_ij = np.power(sim_ij, float(freedom + 1)/2)
    
    elif dist_metrics == 'cosine':
        sim_ij = np.matmul(x, np.transpose(y))
        sim_ij = 1/1(1+ np.exp(-sim_ij))
        # return F.cosine_similarity(x, y.unsqueeze(1), dim=-1)
    if std:
        means = sim_ij.mean(dim=1, keepdim=True)
        stds = sim_ij.std(dim=1, keepdim=True)
        sim_ij = (sim_ij - means) / stds
    if to_distribution:
        sim_ij = (sim_ij.t() / torch.sum(sim_ij, 1)).t() # (n_samples, n_clusters)
    return sim_ij

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    https://datascience.stackexchange.com/questions/17461/how-to-test-accuracy-of-an-unsupervised-clustering-model-output
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    # ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size