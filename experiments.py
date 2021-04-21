from utils import *
from model import TreeNN, FullNN, kfoldvalidation
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
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
    beta = 1
    max_iter = 1000
    num_samples = 4000
    
    np.random.seed(0)    
    A = np.random.randn(num_var, num_var)
    A = (A.T + A)*0.5
    for i in range(num_var):
        for j in range(num_var):
            if abs(j-i) >= 2: A[i, j] = 0
            #else: A[i, j] = 1
    #A = A + 1*np.eye(num_var)
    A = A/np.max(np.max(abs(A)))
    eig, _ = np.linalg.eig(A)
    A = A - np.eye(num_var)*(np.min(eig)/2)
    #A[2,3] = 0
    #A[3,2] = 0
    #print(eig)
    
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
        sample = MHSampling4Ising(np.random.randint(0, 2, size=num_var)*2-1, target_pdf, max_iter)
        samples.append(sample)
        if target_pdf(sample) > 0.4:
            high.append(i)
            #print('>0.4:', sample)
    samples = np.array(samples)
    marginal = marginalize4Ising(samples)

    corr = pearson_corr(samples)

    
    
    '''
    plt.matshow(abs(A))
    plt.show()
    print(corr)
    plt.matshow(abs(corr))
    plt.show()
    '''

    for i in range(num_var): corr[i, i] = 0
    G = nx.from_numpy_matrix(abs(corr), create_using=nx.Graph)
    #layout = nx.spring_layout(G)
    #nx.draw(G, layout)
    #nx.draw_networkx_edge_labels(G, pos=layout)
    #plt.show()
    T = nx.maximum_spanning_tree(G)
    #layout = nx.spring_layout(T)
    #nx.draw(T, layout)
    #nx.draw_networkx_edge_labels(T, pos=layout)
    #plt.show()
    Tedges = sorted(T.edges)
    
    
    node_hidden_layers = [5, 5]
    root_hidden_layers = [5, 5]
    batch_size = 256
    epoches = 300
    lr = 0.01
   

    #Tedges = []
    #for i in range(num_var):
    #    for j in range(i, num_var):
    #        Tedges.append([i, j])
    num_or_size_split = len(Tedges)

    target = np.zeros(num_samples)
    for i in range(num_samples): target[i] = target_pdf(samples[i])
    #print(target)

    plt.figure()
    plt.hist(target, bins=6)
    #plt.show()
    plt.savefig('prob.png')
    
    dist = np.zeros(2**num_var)
    for i in range(2**num_var): dist[i] = target_pdf(allcases[i])
    #print(target)

    plt.figure()
    plt.hist(dist, bins=6)
    #plt.show()
    plt.savefig('dist.png')
    
    '''
    hidden_layers = [num_var, num_var, num_var]
    model1 = FullNN(lr, len(samples[0]), hidden_layers, batch_size)
    model1.fit(samples, target, epoches, batch_size)   
    for i in range(len(high)):
        #print(samples[high[i]])
        print(model1.predict(np.expand_dims(samples[high[i]], 0))) 
    '''

    model = TreeNN(lr, num_or_size_split, node_hidden_layers, root_hidden_layers, batch_size)
    samples_preprocessed = preprocess(samples, Tedges)
    kfoldvalidation(model, samples_preprocessed, target, epoches, batch_size)
    
    
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
    
    plt.figure()
    plt.hist(pred, bins=6)
    #plt.show()
    plt.savefig('pred.png')
    
    plt.figure()
    plt.hist(pred/sum(allcases_pred), bins=6)
    #plt.show()
    plt.savefig('pred_nm.png')
    
if __name__ == "__main__":
    main()
