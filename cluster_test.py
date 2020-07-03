import time
import os
import torch
import pickle
import numpy as np
from statistics import mean
from src.mgm_cluster import mgm_cluster
from src.rrwm import RRWM
from utils.cluster_data_prepare import DataGenerator
from utils.evaluation import evaluate_cluster_accuracy
from utils.hungarian import hungarian
from utils.cfg import cfg as CFG

# set dataset and class for offline multi-graph matching test
dataset_name = "WILLOW-ObjectClass"
# you can remove some classes type during the debug process
class_name = ["Car", "Motorbike", "Face", "Winebottle", "Duck"]

# set parameters for offline multi-graph matching test
test_iter = 1  # test iteration for each class, please set it less than 5 or new data will be generated

# number of graphs, inliers and outliers only affect the generated data (when test_iter is larger than 5),
# these parameters will not be used when the data is loaded from TestPrepare.
test_num_graph = 8  # number of graphs in each test
test_num_cluster = 2  # number of clusters
test_num_graph_cluster = 4  # number of graphs in each dataset(cluster)
test_num_inlier = 10  # number of inliers in each graph
test_num_outlier = 2  # number of outliers in each graph

assert test_num_graph == test_num_cluster * test_num_graph_cluster

rrwm = RRWM()
cfg = CFG()

print("Test begin: test online multi-graph matching on {}".format(dataset_name))
time_list, cp_list, ri_list, ca_list = [], [], [], []
for i_iter in range(test_iter):
    # prepare affinity matrix data for graph matching

    # set the path for loading data
    test_data_folder = 'cluster_data_{}*{}'.format(test_num_cluster, test_num_graph_cluster)
    test_data_folder_path = "data" + "/" + "TestPrepare" + "/" + test_data_folder
    if not os.path.exists(test_data_folder_path):
        os.mkdir(test_data_folder_path)
    test_data_path = "data" + "/" + "TestPrepare" + "/" + test_data_folder + "/" + "test_data_" + str(i_iter)
    if os.path.exists(test_data_path):
        # load data from "/TestPrepare/{test_data_folder_path}/test_data_{i_iter}"
        with open(test_data_path, "rb") as f:
            data = pickle.load(f)
    else:
        # if nothing can be loaded, generate new data and save it
        data = DataGenerator(
            data_path=None,
            num_graphs=test_num_graph,
            num_inlier=test_num_inlier,
            num_outlier=test_num_outlier,
            num_cluster=test_num_cluster,
            num_graphs_cluster=test_num_graph_cluster,
        )
        for i_class in range(test_num_cluster):
            class_path = "data" + "/" + dataset_name + "/" + class_name[i_class]
            data.add_data(
                data_path=class_path,
                num_inlier=test_num_inlier,
                num_outlier=test_num_outlier,
                num_graphs=test_num_graph_cluster
            )
        data.preprocess()
        with open(test_data_path, "wb") as f:
            pickle.dump(data, f)

    # pairwise matching: RRWM

    # set the path for loading pairwise matching results
    init_mat_path = "data" + "/" + "TestPrepare" + "/" + test_data_folder + "/" + "init_mat_" + str(i_iter)
    if os.path.exists(init_mat_path):
        # load pairwise matching results from "/TestPrepare/{ClassType}/init_mat_{i_iter}"
        with open(init_mat_path, "rb") as f:
            X = pickle.load(f)
    else:
        # if nothing can be loaded, generate the initial matching results and save them
        m, n = data.num_graphs, data.num_nodes
        Kt = torch.tensor(data.K).reshape(-1, n * n, n * n).cuda()
        ns_src = torch.ones(m * m).int().cuda() * n
        ns_tgt = torch.ones(m * m).int().cuda() * n
        X_continue = rrwm(Kt, n, ns_src, ns_tgt).reshape(m * m, n, n).transpose(1, 2).contiguous()
        X = hungarian(X_continue, ns_src, ns_tgt).reshape(m, m, n, n).cpu().detach().numpy()
        with open(init_mat_path, "wb") as f:
            pickle.dump(X, f)

    # apply MGM-Floyd to get better matching results
    tic = time.time()
    cluster = mgm_cluster(X, data.K, data.num_graphs, data.num_nodes, data.num_cluster)
    toc = time.time()

    # evaluate the final matching results
    cp, ri, ca = evaluate_cluster_accuracy(data.gt, data.gt, data.num_cluster, data.num_graphs_cluster)
    time_list.append(toc - tic)
    cp_list.append(cp)
    ri_list.append(ri)
    ca_list.append(ca)

    print("Performance on iter{}".format(i_iter))
    print("Clustering Purity: {:.4f}, Rand Index: {:.4f}, Clustering Accuracy: {:.4f}, time: {:.4f}".
          format(cp, ri, ca, toc - tic))

avg_time = mean(time_list)
avg_cp = mean(cp_list)
avg_ri = mean(ri_list)
avg_ca = mean(ca_list)
print("Overall Performance")
print("Clustering Purity: {:.4f}, Rand Index: {:.4f}, Clustering Accuracy: {:.4f}, time: {:.4f}".
      format(avg_cp, avg_ri, avg_ca, avg_time))
