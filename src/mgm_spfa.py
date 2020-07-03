import numpy as np
import torch
from queue import Queue
from collections import deque

def mgm_spfa(K, X, num_graph, num_node, lambd=0.8, alt=""):
    """
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param X: matching results, X[:-1, :-1] is the matching results obtained by last iteration of MGM-SPFA,
              X[num_graph,:] and X[:,num_graph] is obtained via two-graph matching solver(RRWM), We suppose the last
              graph is the new coming graph. (num_graph, num_graph, num_node, num_node)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: X, matching results, match graph_m to {graph_1, ... , graph_m-1)
    """

    if alt == 'slf':
        return mgm_spfa_slf(K,X,num_graph,num_node,lambd)
    elif alt == 'lll':
        return mgm_spfa_lll(K,X,num_graph,num_node,lambd)
    elif alt == 'dfs':
        return mgm_spfa_dfs(K,X,num_graph,num_node,lambd)
    elif alt == 'vec':
        return mgm_spfa_vec(K,X,num_graph,num_node,lambd)

    N = num_graph-1
    q = deque(range(N))
    counter = 0
    while q and counter < num_graph*num_graph:
        x = q.popleft()
        cp_xN = pair_wise_consistency(x,N,X[x,N],X,num_node,num_graph)
        for y in range(num_graph):
            s_org = (1-lambd)*affinity_score(X[y,N],K[y,N],num_node) + lambd*np.sqrt(pair_wise_consistency(y,y,X[y,y],X,num_node,num_graph)*pair_wise_consistency(y,N,X[y,N],X,num_node,num_graph))
            s_opt = (1-lambd)*affinity_score(X[y,x].dot(X[x,N]),K[y,N],num_node) + lambd*np.sqrt(pair_wise_consistency(y,x,X[y,x],X,num_node,num_graph)*cp_xN)
            if s_org < s_opt:
                q.append(y)
                X[y,N] = X[N,y] = X[y,x].dot(X[x,N])
                counter += 1
                if counter % 2 ==0:
                    cp_xN = pair_wise_consistency(x,N,X[x,N],X,num_node,num_graph)

    for x in range(num_graph):
        cp_xN = pair_wise_consistency(x,N,X[x,N],X,num_node,num_graph)
        cp_xx = pair_wise_consistency(x,x,X[x,x],X,num_node,num_graph)
        for y in range(x,num_graph):
            s_org = (1-lambd)*affinity_score(X[x,y],K[x,y],num_node) + lambd*np.sqrt(cp_xx*pair_wise_consistency(x,y,X[x,y],X,num_node,num_graph))
            s_opt = (1-lambd)*affinity_score(X[x,N].dot(X[N,y]),K[x,y],num_node) + lambd*np.sqrt(cp_xN*pair_wise_consistency(N,y,X[N,y],X,num_node,num_graph))
            if s_org < s_opt:
                X[x,y] = X[y,x] = X[x,N].dot(X[N,y])

    return X


def pair_wise_consistency(i,j,x_ij,X,n,m):
	s = 0
	for k in range(m):
		s += np.linalg.norm(x_ij-X[i,k].dot(X[k,j]))/2/n/m
	return 1-s


def affinity_score(X_ij,K_ij,n):
    scr = X_ij.T.reshape(1,-1).dot(K_ij).dot(X_ij.T.reshape(-1,1))
    GT = np.eye(n,n)
    denom = GT.T.reshape(1,-1).dot(K_ij).dot(GT.T.reshape(-1,1))
    return (scr/denom).item()



def mgm_spfa_slf(K, X, num_graph, num_node, lambd=0.8):

    N = num_graph-1
    q = deque(range(N))
    counter = 0
    S = np.zeros(num_graph)
    while q and counter < num_graph*num_graph:
        x = q.popleft()
        cp_xN = pair_wise_consistency(x,N,X[x,N],X,num_node,num_graph)
        for y in range(num_graph):
            s_org = (1-lambd)*affinity_score(X[y,N],K[y,N],num_node) + lambd*np.sqrt(pair_wise_consistency(y,y,X[y,y],X,num_node,num_graph)*pair_wise_consistency(y,N,X[y,N],X,num_node,num_graph))
            S[y] = s_org
            s_opt = (1-lambd)*affinity_score(X[y,x].dot(X[x,N]),K[y,N],num_node) + lambd*np.sqrt(pair_wise_consistency(y,x,X[y,x],X,num_node,num_graph)*cp_xN)
            if s_org < s_opt:
                q.append(y)
                X[y,N] = X[N,y] = X[y,x].dot(X[x,N])
                S[y] = s_opt
                if S[q[0]] < S[q[-1]]:
                    q.rotate(1)
                counter += 1
                if counter % 2 ==0:
                    cp_xN = pair_wise_consistency(x,N,X[x,N],X,num_node,num_graph)

    for x in range(num_graph):
        cp_xN = pair_wise_consistency(x,N,X[x,N],X,num_node,num_graph)
        cp_xx = pair_wise_consistency(x,x,X[x,x],X,num_node,num_graph)
        for y in range(x,num_graph):
            s_org = (1-lambd)*affinity_score(X[x,y],K[x,y],num_node) + lambd*np.sqrt(cp_xx*pair_wise_consistency(x,y,X[x,y],X,num_node,num_graph))
            s_opt = (1-lambd)*affinity_score(X[x,N].dot(X[N,y]),K[x,y],num_node) + lambd*np.sqrt(cp_xN*pair_wise_consistency(N,y,X[N,y],X,num_node,num_graph))
            if s_org < s_opt:
                X[x,y] = X[y,x] = X[x,N].dot(X[N,y])

    return X



def mgm_spfa_lll(K, X, num_graph, num_node, lambd=0.8):

    N = num_graph-1
    q = deque(range(N))
    counter = 0
    S = np.zeros(num_graph)
    while q and counter < num_graph*num_graph:
        x = q.popleft()
        cp_xN = pair_wise_consistency(x,N,X[x,N],X,num_node,num_graph)
        for y in range(num_graph):
            s_org = (1-lambd)*affinity_score(X[y,N],K[y,N],num_node) + lambd*np.sqrt(pair_wise_consistency(y,y,X[y,y],X,num_node,num_graph)*pair_wise_consistency(y,N,X[y,N],X,num_node,num_graph))
            S[y] = s_org
            s_opt = (1-lambd)*affinity_score(X[y,x].dot(X[x,N]),K[y,N],num_node) + lambd*np.sqrt(pair_wise_consistency(y,x,X[y,x],X,num_node,num_graph)*cp_xN)
            if s_org < s_opt:
                q.append(y)
                X[y,N] = X[N,y] = X[y,x].dot(X[x,N])
                S[y] = s_opt
                if S[q].all():
                    avg = S[q].mean()
                    for i,v in enumerate(q):
                        if S[v] >= avg:
                            q.rotate(-i)
                            break
                counter += 1
                if counter % 2 ==0:
                    cp_xN = pair_wise_consistency(x,N,X[x,N],X,num_node,num_graph)

    for x in range(num_graph):
        cp_xN = pair_wise_consistency(x,N,X[x,N],X,num_node,num_graph)
        cp_xx = pair_wise_consistency(x,x,X[x,x],X,num_node,num_graph)
        for y in range(x,num_graph):
            s_org = (1-lambd)*affinity_score(X[x,y],K[x,y],num_node) + lambd*np.sqrt(cp_xx*pair_wise_consistency(x,y,X[x,y],X,num_node,num_graph))
            s_opt = (1-lambd)*affinity_score(X[x,N].dot(X[N,y]),K[x,y],num_node) + lambd*np.sqrt(cp_xN*pair_wise_consistency(N,y,X[N,y],X,num_node,num_graph))
            if s_org < s_opt:
                X[x,y] = X[y,x] = X[x,N].dot(X[N,y])

    return X



def mgm_spfa_dfs(K, X, num_graph, num_node, lambd=0.8):
    N = num_graph-1
    q = deque(range(N))
    counter = 0
    while q and counter < num_graph*num_graph:
        x = q.pop()
        cp_xN = pair_wise_consistency(x,N,X[x,N],X,num_node,num_graph)
        for y in range(num_graph):
            s_org = (1-lambd)*affinity_score(X[y,N],K[y,N],num_node) + lambd*np.sqrt(pair_wise_consistency(y,y,X[y,y],X,num_node,num_graph)*pair_wise_consistency(y,N,X[y,N],X,num_node,num_graph))
            s_opt = (1-lambd)*affinity_score(X[y,x].dot(X[x,N]),K[y,N],num_node) + lambd*np.sqrt(pair_wise_consistency(y,x,X[y,x],X,num_node,num_graph)*cp_xN)
            if s_org < s_opt:
                q.append(y)
                X[y,N] = X[N,y] = X[y,x].dot(X[x,N])
                counter += 1
                if counter % 2 ==0:
                    cp_xN = pair_wise_consistency(x,N,X[x,N],X,num_node,num_graph)

    for x in range(num_graph):
        cp_xN = pair_wise_consistency(x,N,X[x,N],X,num_node,num_graph)
        cp_xx = pair_wise_consistency(x,x,X[x,x],X,num_node,num_graph)
        for y in range(x,num_graph):
            s_org = (1-lambd)*affinity_score(X[x,y],K[x,y],num_node) + lambd*np.sqrt(cp_xx*pair_wise_consistency(x,y,X[x,y],X,num_node,num_graph))
            s_opt = (1-lambd)*affinity_score(X[x,N].dot(X[N,y]),K[x,y],num_node) + lambd*np.sqrt(cp_xN*pair_wise_consistency(N,y,X[N,y],X,num_node,num_graph))
            if s_org < s_opt:
                X[x,y] = X[y,x] = X[x,N].dot(X[N,y])

    return X


def mgm_spfa_vec(K,X,num_graph, num_node, lambd=0.8):

    N = num_graph-1
    K = torch.from_numpy(K).cuda()
    X = torch.from_numpy(X).cuda()
    count = 0
    for i in range(num_graph*num_graph):
        pc = pair_wise_consistency_vec(X,num_node,num_graph)
        s_org = (1-lambd)*affinity_score_vec(X[:,N].view(-1,1,num_node,num_node),K[:,N].view(-1,1,num_node**2,num_node**2),num_node).squeeze(1)+lambd*torch.sqrt(pc.diagonal()*pc[N,:])

        X_opt = X.view(-1,num_node,num_node).bmm(X[N,:].view(1,-1,num_node,num_node).expand(num_graph,-1,-1,-1).reshape(-1,num_node,num_node)).view(X.shape)
        K_opt = K[:,N].view(-1,1,num_node**2,num_node**2).expand(-1,num_graph,-1,-1)
        s_opt = (1-lambd)*affinity_score_vec(X_opt,K_opt,num_node)+lambd*torch.sqrt(pc*pc[N,:].view(1,num_graph).expand(num_graph,-1))
        
        s_opt,inds = s_opt.max(1)
        X_opt = X_opt[range(X_opt.shape[0]),inds]
        X[s_opt>s_org,N] = X[N,s_opt>s_org] = X_opt[s_opt>s_org]
        if not torch.any(s_opt>s_org):
            break

    pc = pair_wise_consistency_vec(X,num_node,num_graph)
    s_org = (1-lambd)*affinity_score_vec(X,K,num_node) + lambd*torch.sqrt(pc.diagonal().view(-1,1).expand(-1,num_graph)*pc)
        
    X_opt = X[:,N].view(-1,1,num_node,num_node).expand(-1,num_graph,-1,-1).reshape(-1,num_node,num_node).bmm(X[N,:].view(1,-1,num_node,num_node).expand(num_graph,-1,-1,-1).reshape(-1,num_node,num_node)).view(X.shape)
    s_opt = (1-lambd)*affinity_score_vec(X_opt,K,num_node) + lambd*torch.sqrt(pc[:,N].view(-1,1).expand(-1,num_graph)*pc[N,:].view(1,-1).expand(num_graph,-1))
    X[s_opt>s_org] = X_opt[s_opt>s_org]

    

    return X.cpu().numpy()



def affinity_score_vec(X,K,n):

    XT = X.transpose(2,3)
    vec_XT = XT.reshape(XT.shape[0]*XT.shape[1],1,-1)
    vec_X = XT.reshape(XT.shape[0]*XT.shape[1],-1,1)
    vec_K = K.reshape(K.shape[0]*K.shape[1],K.shape[2],K.shape[3])
    scr = vec_XT.bmm(vec_K).bmm(vec_X)

    GT = torch.eye(n,n,dtype=vec_K.dtype,device=K.get_device()).view(1,n,n).expand((XT.shape[0]*XT.shape[1],n,n))
    vec_GTT = GT.view(GT.shape[0],1,-1)
    vec_GT = GT.view(GT.shape[0],-1,1)
    denom = vec_GTT.bmm(vec_K).bmm(vec_GT)

    return (scr/denom).view(X.shape[0],X.shape[1])


def pair_wise_consistency_vec(X,n,m):
    m1 = X.view(m,m,1,n,n).expand(-1,-1,m,-1,-1)
    m2 = m1.transpose(0,2)
    m3 = m1.transpose(1,2)
    s = (m3.reshape(-1,n,n) - m1.reshape(-1,n,n).bmm(m2.reshape(-1,n,n))).view(m,m,m,n,n).norm(dim=(3,4)).sum(1).squeeze(1)

    return 1-s/m/n/2