import numpy as np
from statistics import mean


def evaluate_matching_accuracy(mat, mat_gt, num_graph, num_inlier):
    """
    :param mat: matching results, (num_graph, num_graph, num_node, num_node)
    :param mat_gt: matching ground truth, (num_graph, num_graph, num_node, num_node)
    :param num_graph: number of graphs, int
    :param num_inlier: number of inliers, int
    :return: accuracy
    """
    acc_list = []
    for i in range(num_graph):
        for j in range(num_graph):
            x = mat[i, j][:num_inlier, :num_inlier]
            x_gt = mat_gt[i, j][:num_inlier, :num_inlier]
            acc = np.sum(np.sum(np.abs(x - x_gt), 1) == 0) / num_inlier
            acc_list.append(acc)
    return mean(acc_list)


def evaluate_cluster_accuracy(cluster, cluster_gt, num_clusters, num_graphs_cluster):
    """
    :param cluster: an numpy array where cluster[i] tells the cluster g_i belong to, (num_graph, 1)
    :param cluster_gt: ground truth an numpy array where cluster_gt[i] tells the cluster g_i belong to, (num_graph, 1)
    :param num_clusters: number of clusters, int
    :param num_graphs_cluster: number of graphs in each clusters, int
    :return: Clustering Purity, Rand Index, Clustering Accuracy
    """
    assert cluster.shape == cluster_gt.shape, "cluster shape is supposed to be {}, but got {}"\
        .format(cluster_gt.shape, cluster.shape)
    assert cluster.shape[0] == num_graphs_cluster * num_clusters, "clusters first dimension should be {}, but got {}"\
        .format(cluster_gt.shape, cluster.shape)
    print(cluster.reshape(1, -1))
    print(cluster_gt.reshape(1, -1))
    cluster = cluster.astype(int)
    cluster_gt = cluster_gt.astype(int)
    cp = get_cp(cluster, cluster_gt, num_clusters, num_graphs_cluster)
    ri = get_ri(cluster, cluster_gt, num_clusters, num_graphs_cluster)
    ca = get_ca(cluster, cluster_gt, num_clusters, num_graphs_cluster)
    return cp, ri, ca


def get_cp(cluster, cluster_gt, num_clusters, num_graphs_cluster):
    tot = 0
    num_graphs = num_clusters * num_graphs_cluster
    for i in range(num_clusters):
        cnt = np.zeros((num_clusters, 1))
        for j in range(num_graphs):
            if cluster[j] == i:
                cnt[cluster_gt[j]] += 1
        mx = np.max(cnt)
        tot = tot + mx
    cp = tot / num_graphs
    return cp


def get_ri(cluster, cluster_gt, num_clusters, num_graphs_cluster):
    tot = 0
    tp = 0
    num_graphs = num_clusters * num_graphs_cluster
    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            tot = tot + 1
            if (cluster[i] == cluster[j] and cluster_gt[i] == cluster_gt[j]) or \
                    (cluster[i] != cluster[j] and cluster_gt[i] != cluster_gt[j]):
                tp = tp + 1
    ri = tp / tot
    return ri


def get_ca(cluster, cluster_gt, num_clusters, num_graphs_cluster):
    ca = 0
    tot = np.zeros((num_clusters))
    num_graphs = num_clusters * num_graphs_cluster
    for i in range(num_graphs):
        tot[cluster_gt[i]] += 1

    for i_cluster in range(num_clusters):
        cnt1 = np.zeros((num_clusters))
        cnt2 = np.zeros((num_clusters))
        for i in range(num_graphs):
            if cluster_gt[i] == i_cluster:
                cnt1[cluster[i]] += 1
            if cluster[i] == i_cluster:
                cnt2[cluster_gt[i]] += 1
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                if tot[i_cluster] > 0:
                    ca += (cnt1[i] * cnt1[j]) / tot[i_cluster] ** 2
                if tot[i] > 0 and tot[j] > 0:
                    ca += (cnt2[i] * cnt2[j]) / (tot[i] * tot[j])

    ca = 1 - ca / num_clusters
    return ca
