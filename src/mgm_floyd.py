import numpy as np

def mgm_floyd(X, K, num_graph, num_node):
    """
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: matching results, (num_graph, num_graph, num_node, num_node)
    """
    
    for lambd in [0,0.5]:
        for v in range(num_graph):
            for x in range(num_graph):
                cp_xv = pair_wise_consistency(x,v,X,num_node,num_graph)
                for y in range(x+1,num_graph):
                    s_org = (1-lambd)*affinity_score(X[x,y],K[x,y],num_node)
                    if lambd > 0:
                        s_org += lambd*pair_wise_consistency(x,y,X,num_node,num_graph)
                    s_opt = (1-lambd)*affinity_score(X[x,v].dot(X[v,y]),K[x,y],num_node)
                    if lambd > 0:
                        s_opt += lambd*np.sqrt(cp_xv*pair_wise_consistency(v,y,X,num_node,num_graph))
                    if s_org < s_opt:
                        X[x,y] = X[y,x] = X[x,v].dot(X[v,y])
                        cp_xv = pair_wise_consistency(x,v,X,num_node,num_graph)
    return X

def pair_wise_consistency(i,j,X,n,m):
	s = 0
	for k in range(m):
		s += np.linalg.norm(X[i,j]-X[i,k].dot(X[k,j]))/2/n/m
	return 1-s



def affinity_score(X_ij,K_ij,n):
    scr = X_ij.T.reshape(1,-1).dot(K_ij).dot(X_ij.T.reshape(-1,1))
    GT = np.eye(n,n)
    denom = GT.T.reshape(1,-1).dot(K_ij).dot(GT.T.reshape(-1,1))
    return (scr/denom).item()