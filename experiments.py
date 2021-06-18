from utils import *
from model import NNModel, kfoldvalidation, rekfoldvalidation
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def sigmoid(x):
    return 1/(1+np.exp(-x))

def kldivergence(p, q):
    return sum(-p*np.log(q) + p*np.log(p))
    
def jsdivergence(p, q):
    m = (p+q)*0.5
    return kldivergence(p, m)*0.5 + kldivergence(q, m)*0.5

def main():
    num_var = 10
    what = 'LN'
    beta = 0.1
    betastr = str(0.1).replace('.', '-') + '_' + what
    max_iter = 1000
    num_samples = 10000
    rate = 0.3
    
    np.random.seed(0)    
    A = np.random.randn(num_var, num_var)
    A = (A.T + A)*0.5
    for i in range(num_var):
        for j in range(num_var):
            if abs(j-i) != 1: A[i, j] = 0
            else: A[i, j] = -1
    #A = A + 1*np.eye(num_var)
    A = A/np.max(np.max(abs(A)))
    eig, _ = np.linalg.eig(A)
    A = A - np.eye(num_var)*(np.min(eig))
    # eig, _ = np.linalg.eig(A)
    #A[2,3] = 0
    #A[3,2] = 0
    # print(eig)

    
    #print(A[0:10,0:10])
    E = lambda x: E_linear(A, x)
    target_pdf = lambda x: boltzmann_pdf(beta, E, x, 1)
    allcases = np.zeros((2**num_var, num_var))
    enumerate(allcases, num_var)
    #print(allcases)
    const = 0
    for i in range(len(allcases)):
        const += target_pdf(allcases[i,:])


    print('const=', const)
    target_pdf = lambda x: boltzmann_pdf(beta, E, x, 1/const)


    #print(E(np.random.randint(0, 2, size=num_var)*2-1))

    samples = []
    high = []
    for i in range(num_samples):
        sample = MHSampling4Ising(np.random.randint(0, 2, size=num_var)*2-1, target_pdf, max_iter, rate)
        samples.append(sample)
    samples = np.array(samples)

    target = np.zeros(num_samples)
    for i in range(num_samples): 
        target[i] = target_pdf(samples[i])
    '''
    high = np.tile(high, 10)
    print(len(high), type(high))
    idx = list(range(num_samples)) + list(high)
    samples = samples[idx, :]
    target = target[idx]
    num_samples = len(target)
    idx = list(range(num_samples))
    np.random.shuffle(idx)
    samples = samples[idx, :]
    target = target[idx]
    '''

    plt.figure()
    plt.hist(target, bins=10)
    #plt.show()
    plt.savefig('prob' + '_' + '%2d' % num_var  +  '_' + betastr + '.png')
    
    dist = np.zeros(2**num_var)
    for i in range(2**num_var): dist[i] = target_pdf(allcases[i])
    #print(target)

    plt.figure()
    plt.hist(dist, bins=10)
    #plt.show()
    plt.savefig('dist' + '_' + '%2d' % num_var  +  '_' + betastr + '.png')

    labels, freq = frequency(samples)
    dist_em = probbyfreq(allcases, freq)
    print('cumulative freq', sum(freq))
    print(sum(dist), sum(dist_em))

    plt.figure()
    plt.hist(dist_em, bins=10)
    #plt.show()
    plt.savefig('dist_em' + '_' + '%2d' % num_var  +  '_' + betastr + '.png')

    plt.figure()
    plt.hist(labels, bins=10)
    plt.savefig('label' + '_' + '%2d' % num_var  +  '_' + betastr + '.png')
    
    marginal_em = marginalize4Ising(samples)
    marginal_th = marginalize4Ising_truth(allcases, dist)
    np.save('margin_em' + '_' + '%2d' % num_var  +  '_' + betastr + '.npy', marginal_em)
    np.save('margin_th' + '_' + '%2d' % num_var  +  '_' + betastr + '.npy', marginal_th)
    ep_th = expectation_truth(allcases, dist)

    corr = pearson_corr(samples)
    corr_th = pearson_corr_truth(marginal_th, ep_th)

    S = abs(corr)
    print(S.shape, num_var)
    D = np.zeros((num_var, num_var))
    Dh = np.zeros((num_var, num_var))
    D_h = np.zeros((num_var, num_var))
    for i in range(num_var): 
        D[i, i] = np.sum(S[i, :])
        Dh[i, i] = np.sqrt(D[i, i])
        D_h[i, i] = 1/np.sqrt(D[i, i])
    L = D - S
    LN = np.matmul(np.matmul(D_h, L), D_h)
    v = np.matmul(Dh, np.ones((num_var, 1)))
    E = np.eye(num_var) - np.matmul(v, v.T)
    M = np.matmul(np.matmul(E, LN), E)
    eigLN, vecLN = np.linalg.eig(LN)
    eigM, vecM = np.linalg.eig(M)
    print('eigLN: ', eigLN)
    # print('eigM: ', eigM)
    
    abseigLN = abs(eigLN)
    abseigM = abs(eigM)
    mineigLN = sorted(range(num_var), key=lambda x: abseigLN[x])[1]
    mineigM = sorted(range(num_var), key=lambda x: abseigM[x])[1]
    
    print(mineigLN, mineigM)
    print(abseigLN[mineigLN], abseigM[mineigM])
    eigvecLN = vecLN[:, mineigLN]
    eigvecM = vecM[:, mineigM]
    print(eigvecLN)
    print(eigvecM)
    clustersLN = clusters(eigvecLN, 2)
    clustersM = clusters(eigvecM, 2)
    print(clustersLN)
    print(clustersM)
    '''
    print('correlation matrix by sampling:')
    print(corr)
    '''
    plt.figure()
    plt.matshow(abs(corr))
    plt.savefig('corr_sampling' + '_' + '%2d' % num_var  + '_' + betastr + '.png')

    # corr_th = pearson_corr_truth(marginal_th, ep_th)
    '''
    print('correlation matrix ground truth:')
    print(corr_th) 
    '''
    plt.figure()
    plt.matshow(abs(corr_th))
    plt.savefig('corr_truth' + '_' + '%2d' % num_var  + '_' + betastr + '.png')
    '''
    print('relative error between two correlation matrices:', np.linalg.norm(corr-corr_th)/np.linalg.norm(corr_th))
    '''
    for i in range(num_var): corr[i, i] = 0
    G = nx.from_numpy_matrix(abs(corr), create_using=nx.Graph)
    #layout = nx.spring_layout(G)
    #nx.draw(G, layout)
    #nx.draw_networkx_edge_labels(G, pos=layout)
    #plt.show()
    T = nx.maximum_spanning_tree(G)
    layout = nx.spring_layout(T)
    nx.draw(T, layout)
    nx.draw_networkx_edge_labels(T, pos=layout)
    plt.savefig('tree' + '_' + '%2d' % num_var  + '_' + betastr + '.png')
    #plt.show()
    # Tedges = sorted(T.edges)
    if what == 'LN':
       Tedges = clustersLN
    elif what == 'M':
       Tedges = clustersM
    print('clusters = ', Tedges)
    
    params = {}
    batch_size = 128
    epoches = 100
    params['node_hidden_layers'] = [4, 4]
    params['root_hidden_layers'] = [num_var-1, num_var-1]
    params['batch_size'] = batch_size
    params['lr'] = 0.000001
   

    #Tedges = []
    #for i in range(num_var):
    #    for j in range(i, num_var):
    #        Tedges.append([i, j])
    # params['num_or_size_split'] = len(Tedges)
    params['num_or_size_split'] = [len(Tedges[i]) for i in range(len(Tedges))]
    assert int(sum(params['num_or_size_split'])) == num_var
    

    kl = kldivergence(dist_em, dist)
    print('kl divergence between ground truth and empirical truth', kl)
    js = jsdivergence(dist_em, dist)
    print('js divergence between ground truth and empirical truth', js)
    
    
    print('distance between margin_em and margin_th', np.linalg.norm(marginal_em-marginal_th)/np.linalg.norm(marginal_th))


    ############## TreeNN
    
    # print('############################### Tree NN ############################################')
    # model = NNModel('TreeNN', params)
    # samples_preprocessed = preprocess(samples, Tedges)
    # kfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)
    # rekfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)
    # rekfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)
    # rekfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)
    # rekfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)
    # rekfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)
    # rekfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)
    # rekfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)
    # #rekfoldvalidation(model, preprocess(allcases, Tedges), dist, epoches, batch_size, 5)
    # #rekfoldvalidation(model, preprocess(allcases, Tedges), dist, epoches, batch_size, 5)
    # #rekfoldvalidation(model, preprocess(allcases, Tedges), dist, epoches, batch_size, 5)
    # #rekfoldvalidation(model, samples_preprocessed, labels, epoches, batch_size, 5)
    # #rekfoldvalidation(model, samples_preprocessed, labels, epoches, batch_size, 5)
    # #rekfoldvalidation(model, samples_preprocessed, labels, epoches, batch_size, 5)
    


    # batches = num_samples // batch_size
    # pred = np.zeros(batches*batch_size)
    # for i in range(batches):
    #     #print(samples_preprocessed[high[i]])
    #     #pred.append(model.predict(np.expand_dims(samples_preprocessed[], 0)))
    #     #print(model.predict(samples_preprocessed[i*batch_size:(i+1)*batch_size]))
    #     pred[i*batch_size:(i+1)*batch_size] = model.predict(samples_preprocessed[i*batch_size:(i+1)*batch_size])
    
    
    # sums = np.linalg.norm(target[:len(pred)]-pred)**2
    # print('relative l2 loss =', sums/np.linalg.norm(target[:len(pred)])**2)
    
    
    # allcases_pre = preprocess(allcases, Tedges)
    # batches = int(np.ceil(2**num_var / batch_size))
    # allcases_pred = np.zeros(2**num_var)
    # kl_b = 0.0
    # js_b = 0.0
    # for i in range(batches):
    #     allcases_pred[i*batch_size:min(((i+1)*batch_size, 2**num_var))] = model.predict(allcases_pre[i*batch_size:min(((i+1)*batch_size, 2**num_var)), :])
    #     allcases_i = allcases_pred[i*batch_size:min(((i+1)*batch_size, 2**num_var))]
    #     kl_b += kldivergence(allcases_i, dist[i*batch_size:min(((i+1)*batch_size, 2**num_var))])
    #     js_b += jsdivergence(allcases_i, dist[i*batch_size:min(((i+1)*batch_size, 2**num_var))])
    # print('cumulative dist  =', sum(allcases_pred))
    # kl = kldivergence(allcases_pred, dist)
    # print('kl divergence & its square =', kl, kl**2)
    # js = jsdivergence(allcases_pred, dist)
    # print('js divergence & its square =', js, js**2)
    # kl_b, js_b = kl_b/batches, js_b/batches
    # print('kl divergence & its square(batch) =', kl_b, kl_b**2)
    # print('js divergence & its square(batch) =', js_b, js_b**2)
    
    # marginal_pd = marginalize4Ising_truth(allcases, allcases_pred/sum(allcases_pred))
    # np.save('margin_pdt' + '_' + '%2d' % num_var  +  '_' + betastr + '.npy', marginal_pd)
    # print('distance between margin_pd and margin_th', np.linalg.norm(marginal_pd-marginal_th)/np.linalg.norm(marginal_th))
            
    '''
    plt.figure()
    plt.hist(allcases_pred/sum(allcases_pred), bins=10)
    #plt.show()
    plt.savefig('pred' + '_tree' + '_' + '%2d' % num_var  +  '_' + betastr + '.png')
    '''
    '''
    rekfoldvalidation(model, samples_preprocessed, labels, epoches, batch_size, 5)

    allcases_pre = preprocess(allcases, Tedges)
    batches = int(np.ceil(2**num_var / batch_size))
    allcases_pred = np.zeros(2**num_var)
    kl_b = 0.0
    js_b = 0.0
    for i in range(batches):
        allcases_pred[i*batch_size:min(((i+1)*batch_size, 2**num_var))] = model.predict(allcases_pre[i*batch_size:min(((i+1)*batch_size, 2**num_var)), :])
        allcases_i = allcases_pred[i*batch_size:min(((i+1)*batch_size, 2**num_var))]
        kl_b += kldivergence(allcases_i, dist[i*batch_size:min(((i+1)*batch_size, 2**num_var))])
        js_b += jsdivergence(allcases_i, dist[i*batch_size:min(((i+1)*batch_size, 2**num_var))])
    print('cumulative dist  =', sum(allcases_pred))
    kl = kldivergence(allcases_pred, dist)
    print('kl divergence & its square =', kl, kl**2)
    js = jsdivergence(allcases_pred, dist)
    print('js divergence & its square =', js, js**2)
    kl_b, js_b = kl_b/batches, js_b/batches
    print('kl divergence & its square(batch) =', kl_b, kl_b**2)
    print('js divergence & its square(batch) =', js_b, js_b**2)

    marginal_pd = marginalize4Ising_truth(allcases, allcases_pred/sum(allcases_pred))
    print('distance between margin_pd and margin_th', np.linalg.norm(marginal_pd-marginal_th)/np.linalg.norm(marginal_th))
     

    plt.figure()
    plt.hist(allcases_pred/sum(allcases_pred), bins=10)
    #plt.show()
    plt.savefig('pred_em' + '_tree'  + '_' + '%2d' % num_var  +  '_' + betastr + '.png')
    '''
   



    ############### DenseNN
    
    print('################################# Dense NN ########################################')
    Tedges = []
    for i in range(num_var):
        for j in range(i+1, num_var):
            Tedges.append((i, j))
    
    Tedges = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    
    params['hidden_layers'] = [num_var]
    params['dim_input'] = 2*len(Tedges)
    params['lr'] = 0.05
    samples_preprocessed = preprocess(samples, Tedges)
    model = NNModel('DenseNN', params)
    kfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)
    rekfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)
    rekfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)
    rekfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)
    rekfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)
    rekfoldvalidation(model, preprocess(samples, Tedges), target, epoches, batch_size, 5)

    weights = model.get_weights()
    for name in weights:
        if len(weights[name].shape) == 2:
            plt.figure()
            plt.matshow(weights[name])
            plt.savefig(name+'.png')

    batches = num_samples // batch_size
    pred = np.zeros(batches*batch_size)
    for i in range(batches):
        #print(samples_preprocessed[high[i]])
        #pred.append(model.predict(np.expand_dims(samples_preprocessed[], 0)))
        #print(model.predict(samples_preprocessed[i*batch_size:(i+1)*batch_size]))
        pred[i*batch_size:(i+1)*batch_size] = model.predict(samples_preprocessed[i*batch_size:(i+1)*batch_size])
    
    
    sums = np.linalg.norm(target[:len(pred)]-pred)**2
    print('relative l2 loss =', sums/np.linalg.norm(target[:len(pred)])**2)
    
    
    allcases_pre = preprocess(allcases, Tedges)
    batches = int(np.ceil(2**num_var / batch_size))
    allcases_pred = np.zeros(2**num_var)
    kl_b = 0.0
    js_b = 0.0
    for i in range(batches):
        allcases_pred[i*batch_size:min(((i+1)*batch_size, 2**num_var))] = model.predict(allcases_pre[i*batch_size:min(((i+1)*batch_size, 2**num_var)), :])
        allcases_i = allcases_pred[i*batch_size:min(((i+1)*batch_size, 2**num_var))]
        kl_b += kldivergence(allcases_i, dist[i*batch_size:min(((i+1)*batch_size, 2**num_var))])
        js_b += jsdivergence(allcases_i, dist[i*batch_size:min(((i+1)*batch_size, 2**num_var))])
    print('cumulative dist  =', sum(allcases_pred))
    kl = kldivergence(allcases_pred, dist)
    print('kl divergence & its square =', kl, kl**2)
    js = jsdivergence(allcases_pred, dist)
    print('js divergence & its square =', js, js**2)
    kl_b, js_b = kl_b/batches, js_b/batches
    print('kl divergence & its square(batch) =', kl_b, kl_b**2)
    print('js divergence & its square(batch) =', js_b, js_b**2)
    
    marginal_pd = marginalize4Ising_truth(allcases, allcases_pred/sum(allcases_pred))
    np.save('margin_pdf' + '_' + '%2d' % num_var  +  '_' + betastr + '.npy', marginal_pd)
    print('distance between margin_pd and margin_th', np.linalg.norm(marginal_pd-marginal_th)/np.linalg.norm(marginal_th))
            
    
    plt.figure()
    plt.hist(allcases_pred/sum(allcases_pred), bins=10)
    #plt.show()
    plt.savefig('pred' + '_dense' + '_' + '%2d' % num_var  +  '_' + betastr + '.png')

    '''
    rekfoldvalidation(model, samples_preprocessed, labels, epoches, batch_size, 5)

    allcases_pre = preprocess(allcases, Tedges)
    batches = int(np.ceil(2**num_var / batch_size))
    allcases_pred = np.zeros(2**num_var)
    kl_b = 0.0
    js_b = 0.0
    for i in range(batches):
        allcases_pred[i*batch_size:min(((i+1)*batch_size, 2**num_var))] = model.predict(allcases_pre[i*batch_size:min(((i+1)*batch_size, 2**num_var)), :])
        allcases_i = allcases_pred[i*batch_size:min(((i+1)*batch_size, 2**num_var))]
        kl_b += kldivergence(allcases_i, dist[i*batch_size:min(((i+1)*batch_size, 2**num_var))])
        js_b += jsdivergence(allcases_i, dist[i*batch_size:min(((i+1)*batch_size, 2**num_var))])
    print('cumulative dist  =', sum(allcases_pred))
    kl = kldivergence(allcases_pred, dist)
    print('kl divergence & its square =', kl, kl**2)
    js = jsdivergence(allcases_pred, dist)
    print('js divergence & its square =', js, js**2)
    kl_b, js_b = kl_b/batches, js_b/batches
    print('kl divergence & its square(batch) =', kl_b, kl_b**2)
    print('js divergence & its square(batch) =', js_b, js_b**2)

    marginal_pd = marginalize4Ising_truth(allcases, allcases_pred/sum(allcases_pred))
    print('distance between margin_pd and margin_th', np.linalg.norm(marginal_pd-marginal_th)/np.linalg.norm(marginal_th))
     

    plt.figure()
    plt.hist(allcases_pred/sum(allcases_pred), bins=10)
    #plt.show()
    plt.savefig('pred_em' + '_dense'  + '_' + '%2d' % num_var  +  '_' + betastr + '.png') 
    '''





    
    ############### CNN
    
    # print('################################ CNN  ##############################################')
    # params['filters'] = 1
    # params['kernel_size'] = (1,4)
    # params['num_layers'] = 3
    # params['activation'] = 'sigmoid'
    # params['lr'] = 0.0001
    # params['input_shape'] = (None, 1, num_var, 1)
    # samples_preprocessed = np.zeros((num_samples, 1, num_var, 1))
    # for i in range(num_samples): samples_preprocessed[i, 0, :, 0] = samples[i]
    # model = NNModel('CNN', params)
    # kfoldvalidation(model, samples_preprocessed, target, epoches, batch_size, 5)
    
    


    # batches = num_samples // batch_size
    # pred = np.zeros(batches*batch_size)
    # for i in range(batches):
    #     #print(samples_preprocessed[high[i]])
    #     #pred.append(model.predict(np.expand_dims(samples_preprocessed[], 0)))
    #     #print(model.predict(samples_preprocessed[i*batch_size:(i+1)*batch_size]))
    #     pred[i*batch_size:(i+1)*batch_size] = model.predict(samples_preprocessed[i*batch_size:(i+1)*batch_size])
    
    
    # sums = np.linalg.norm(target[:len(pred)]-pred)**2
    # print('relative l2 loss =', sums/np.linalg.norm(target[:len(pred)])**2)
    
    
    # #allcases_pre = preprocess(allcases, Tedges)
    # allcases_pre = np.zeros((2**num_var, 1, num_var, 1))
    # for i in range(num_samples): allcases_pre[i, 0, :, 0] = allcases[i]
    # batches = int(np.ceil(2**num_var / batch_size))
    # allcases_pred = np.zeros(2**num_var)
    # kl_b = 0.0
    # js_b = 0.0
    # for i in range(batches):
    #     allcases_pred[i*batch_size:min(((i+1)*batch_size, 2**num_var))] = model.predict(allcases_pre[i*batch_size:min(((i+1)*batch_size, 2**num_var)), :, :, :])
    #     allcases_i = allcases_pred[i*batch_size:min(((i+1)*batch_size, 2**num_var))]
    #     kl_b += kldivergence(allcases_i, dist[i*batch_size:min(((i+1)*batch_size, 2**num_var))])
    #     js_b += jsdivergence(allcases_i, dist[i*batch_size:min(((i+1)*batch_size, 2**num_var))])
    # print('cumulative dist  =', sum(allcases_pred))
    # kl = kldivergence(allcases_pred, dist)
    # print('kl divergence & its square =', kl, kl**2)
    # js = jsdivergence(allcases_pred, dist)
    # print('js divergence & its square =', js, js**2)
    # kl_b, js_b = kl_b/batches, js_b/batches
    # print('kl divergence & its square(batch) =', kl_b, kl_b**2)
    # print('js divergence & its square(batch) =', js_b, js_b**2)
    
    # marginal_pd = marginalize4Ising_truth(allcases, allcases_pred/sum(allcases_pred))
    # print('distance between margin_pd and margin_th', np.linalg.norm(marginal_pd-marginal_th)/np.linalg.norm(marginal_th))
            
    
    # plt.figure()
    # plt.hist(allcases_pred/sum(allcases_pred), bins=10)
    # #plt.show()
    # plt.savefig('pred' + '_cnn' + '_' + '%2d' % num_var  +  '_' + betastr + '.png')

    '''
    rekfoldvalidation(model, samples_preprocessed, labels, epoches, batch_size, 5)

    #allcases_pre = preprocess(allcases, Tedges)
    allcases_pre = np.zeros((2**num_var, 1, num_var, 1))
    for i in range(num_samples): allcases_pre[i, 0, :, 0] = allcases[i]
    batches = int(np.ceil(2**num_var / batch_size))
    allcases_pred = np.zeros(2**num_var)
    kl_b = 0.0
    js_b = 0.0
    for i in range(batches):
        allcases_pred[i*batch_size:min(((i+1)*batch_size, 2**num_var))] = model.predict(allcases_pre[i*batch_size:min(((i+1)*batch_size, 2**num_var)), :, :, :])
        allcases_i = allcases_pred[i*batch_size:min(((i+1)*batch_size, 2**num_var))]
        kl_b += kldivergence(allcases_i, dist[i*batch_size:min(((i+1)*batch_size, 2**num_var))])
        js_b += jsdivergence(allcases_i, dist[i*batch_size:min(((i+1)*batch_size, 2**num_var))])
    print('cumulative dist  =', sum(allcases_pred))
    kl = kldivergence(allcases_pred, dist)
    print('kl divergence & its square =', kl, kl**2)
    js = jsdivergence(allcases_pred, dist)
    print('js divergence & its square =', js, js**2)
    kl_b, js_b = kl_b/batches, js_b/batches
    print('kl divergence & its square(batch) =', kl_b, kl_b**2)
    print('js divergence & its square(batch) =', js_b, js_b**2)

    marginal_pd = marginalize4Ising_truth(allcases, allcases_pred/sum(allcases_pred))
    print('distance between margin_pd and margin_th', np.linalg.norm(marginal_pd-marginal_th)/np.linalg.norm(marginal_th))
     

    plt.figure()
    plt.hist(allcases_pred/sum(allcases_pred), bins=10)
    #plt.show()
    plt.savefig('pred_em' + '_cnn'  + '_' + '%2d' % num_var  +  '_' + betastr + '.png')  
    '''
if __name__ == "__main__":
    main()
