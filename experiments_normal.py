from utils import *
from model import TreeNN, FullNN
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def main():
    num_var = 10
    beta = 1
    max_iter = 1000
    num_samples = 3000
    
    A = np.random.rand(num_var, num_var)
    A = (A.T + A)*0.5
    for i in range(num_var):
        for j in range(num_var):
            if abs(j-i) >= 2: A[i, j] = 0
            #else: A[i, j] = 1
    A = A + np.diag(np.diag(A))
    eig, _ = np.linalg.eig(A)
    #A[2,3] = 0
    #A[3,2] = 0
    #print(eig)
    A = A - np.eye(num_var)*eig[np.argmin(eig)] + 0.1
    A = A/np.max(np.max(abs(A)))
    mu = np.random.rand(num_var)/2
    #print(A[0:10,0:10])

    target_pdf = lambda x: gaussian_pdf(mu, A, x)



    #print(E(np.random.randint(0, 2, size=num_var)*2-1))

    samples = []
    high = []
    for i in range(num_samples):
        sample = MHSampling(np.random.rand(num_var), target_pdf, max_iter)
        samples.append(sample)
        if target_pdf(sample) > 0.8:
            high.append(i)
            #print('>0.4:', sample)
    samples = np.array(samples)
    #marginal = marginalize4Ising(samples)
    print("Sampling done!")
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
    layout = nx.spring_layout(T)
    nx.draw(T, layout)
    nx.draw_networkx_edge_labels(T, pos=layout)
    #plt.show()
    Tedges = sorted(T.edges)
    
    
    node_hidden_layers = [2, 2]
    root_hidden_layers = [5]
    batch_size = 24
    epoches = 5
    lr = 0.1

    Tedges = []
    for i in range(num_var):
        for j in range(i, num_var):
            Tedges.append([i, j])
    num_or_size_split = len(Tedges)

    target = np.zeros(num_samples)
    for i in range(num_samples): target[i] = target_pdf(samples[i])
    #print(target)

    hidden_layers = [num_var, num_var, num_var]
    model1 = FullNN(lr, len(samples[0]), hidden_layers, batch_size)
    model1.fit(samples, target, epoches, batch_size)   
    for i in range(len(high)):
        print(samples[high[i]])
        print(model1.predict(np.expand_dims(samples[high[i]], 0))) 

    model = TreeNN(lr, num_or_size_split, node_hidden_layers, root_hidden_layers, batch_size)
    samples_preprocessed = preprocess(samples, Tedges)
    model.fit(samples_preprocessed, target, epoches, batch_size)
    
    for i in range(len(high)):
        print(samples_preprocessed[high[i]])
        print(model.predict(np.expand_dims(samples_preprocessed[high[i]], 0)))
    
if __name__ == "__main__":
    main()