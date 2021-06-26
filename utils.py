## implementation of Metropolis–Hastings sampling
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import sklearn.preprocessing as sp

def clusters(X, levels):
    ans = [np.array(range(len(X)))]
    for i in range(levels):
        ansnew = []
        for j in range(len(ans)):
            kmeans = KMeans(n_clusters=2, random_state=0).fit(np.expand_dims(X[ans[j]], axis=1))
            # print(kmeans.labels_)
            ansnew.append(ans[j][np.where(kmeans.labels_==0)[0]])
            ansnew.append(ans[j][np.where(kmeans.labels_==1)[0]])
        ans = ansnew
    return ans

        

def gaussian_pdf(mean, cov, x):
    k = len(mean)
    assert k == len(x), 'Input dimension is not correct!'
    invcov = np.linalg.solve(cov, np.eye(k))
    diff = x - mean
    prod = diff.T.dot(invcov).dot(diff)
    det = np.linalg.det(cov)
    pdf = 1.0/np.sqrt((2*np.pi)**k / det)*np.exp(-prod*0.5)
    return pdf

def MHSampling(init_state, target_pdf, iter):
    x0 = init_state
    cov = np.eye(len(init_state))
    for t in range(iter):
        x = np.random.multivariate_normal(x0, cov)
        a = target_pdf(x)*gaussian_pdf(x, cov, x0)
        a0 = target_pdf(x0)*gaussian_pdf(x0, cov, x)
        #print(a, a0)
        acceptance = min(1, a/a0)
        u = np.random.rand(1)
        if u < acceptance: x0 = x
    return x0

def uniformlychange1ele(x):
    n = len(x)
    seeds = np.random.rand(0, n, int(n/3))
    for i in range(len(seeds)):
        ind = seeds[i]
        x[ind] = -x[ind]
    return x

def proposaldist(x, y, rate=0.5):
    prob = 1.0
    for i in range(len(x)):
        if abs(x[i]-y[i]) < 1e-14: prob = prob*(1-rate)
        else: prob = prob*rate
    return prob

def uniformlychange(x, rate=0.5):
    y = np.zeros(len(x))
    for i in range(len(x)):
        if np.random.rand() < rate: y[i] = -x[i]
        else: y[i] = x[i]
    return y

def MHSampling4Ising(init_state, target_pdf, iter, rate):
    x0 = init_state
    for t in range(iter):
        x = uniformlychange(x0, rate)
        #acceptance = min(1, target_pdf(x)*proposaldist(x, x0, rate)/target_pdf(x0)*proposaldist(x0, x, rate))
        acceptance = min(1, target_pdf(x)/target_pdf(x0))
        u = np.random.rand(1)
        if u < acceptance: x0 = x
    return x0

def E_linear(A, x):
    return x.T.dot(A).dot(x)

def boltzmann_pdf(beta, E_fn, x, const):
    return const*np.exp(-beta*E_fn(x))

def normal_pdf(mu, A, x):
    k = len(mu)
    diff = x - mu
    assert len(diff.shape) == 1
    invA = np.linalg.solve(A, np.eye(k))
    nu = np.exp(-diff.T.dot(invA).dot(diff)*0.5)
    de = np.sqrt(np.linalg.det(A)*((2*np.pi)**k) )
    #print(nu, de)
    return nu/de

def enumerate(ans, num_var):
    if num_var > 1:
        n = len(ans)//2
        assert n*2 == 2**num_var
        ans[:n, 0] = 1
        enumerate(ans[:n, 1:], num_var-1)
        ans[n:, 0] = -1
        enumerate(ans[n:, 1:], num_var-1)
    else:
        ans[0, :] = 1
        ans[1, :] = -1
        return 


def pearson_corr(samples):
    num_samples = len(samples)
    num_var = len(samples[0])
    corr = np.zeros((num_var, num_var))
    #print(samples.shape, corr.shape)
    for i in range(num_var):
        for j in range(i, num_var):
            corr[i, j] = pearsonr(samples[:, i], samples[:, j])[0]
    return corr.T + corr - np.diag(np.diag(corr))

def simplecount(x0, x1):
    num = len(x0)
    ans = np.zeros((2,2))
    for i in range(num):
        if abs(x0[i]-1) < 1e-14 and abs(x1[i]-1) < 1e-14:
            ans[0, 0] += 1
        elif abs(x0[i]-1) < 1e-14 and abs(x1[i]+1) < 1e-14:
            ans[0, 1] += 1
        elif abs(x0[i]+1) < 1e-14 and abs(x1[i]-1) < 1e-14:
            ans[1, 0] += 1
        else:
            ans[1, 1] += 1
    assert np.sum(np.sum(ans)) == num
    return ans/num

def marginalize4Ising(samples):
    num_var = samples.shape[1]
    marginal = np.zeros((num_var, num_var, 2, 2))
    for i in range(num_var):
        for j in range(i, num_var):
            marginal[i, j, :, :] = simplecount(samples[:, i], samples[:, j])
    return marginal

def marginalize4Ising_truth(samples, dist):
    num_var = samples.shape[1]
    marginal = np.zeros((num_var, num_var, 2, 2))
    for i in range(num_var):
        for j in range(num_var):
            marginal[i, j, :, :] = simplecount_prob(samples[:, i], samples[:, j], dist)
    return marginal

def expectation_truth(samples, dist):
    (m, n) = samples.shape
    ep = np.zeros(n)
    for i in range(n):
        prob1 = 0
        prob_1 = 0
        for j in range(m):
            if abs(samples[j, i]-1) < 1e-14: prob1 += dist[j]
            else: prob_1 += dist[j]
        assert abs(prob1 + prob_1 - 1) < 1e-12
        ep[i] = prob1-prob_1
    return ep

def pearson_corr_truth(margin, exp):
    n = len(exp)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            epi, epj = exp[i], exp[j]
            cov = margin[i, j, 0, 0] + margin[i, j, 1, 1] - margin[i, j, 0, 1]- margin[i, j, 1, 0] - epi*epj
            sigmai = np.sqrt(1-epi**2)
            sigmaj = np.sqrt(1-epj**2)
            corr[i, j] = cov/sigmai/sigmaj
    corr = corr + corr.T - np.diag(np.diag(corr))
    return corr


def simplecount_prob(x0, x1, dist):
    num = len(x0)
    ans = np.zeros((2,2))
    for i in range(num):
        if abs(x0[i]-1) < 1e-14 and abs(x1[i]-1) < 1e-14:
            ans[0, 0] += dist[i]
        elif abs(x0[i]-1) < 1e-14 and abs(x1[i]+1) < 1e-14:
            ans[0, 1] += dist[i]
        elif abs(x0[i]+1) < 1e-14 and abs(x1[i]-1) < 1e-14:
            ans[1, 0] += dist[i]
        else:
            ans[1, 1] += dist[i]
    #assert abs(np.sum(np.sum(ans))-1) < 1e-14
    return ans



def preprocess(samples, MST):
    edges = sum([len(MST[i]) for i in range(len(MST))])
    num_samples = samples.shape[0]
    ans = np.zeros((num_samples, edges))
    count = 0
    for i in range(len(MST)):
        edge = MST[i]
        for j in range(len(edge)):
            ans[:, count+j] = samples[:, edge[j]]
        count += len(edge)
    return ans

def to0or1(num):
    return np.array((num+1) / 2)

def frequency(samples):
    (n, m) = samples.shape
    freq = np.zeros(2**m)
    weights = np.array([2**(m-i-1) for i in range(m)])
    idx = to0or1(samples).dot(weights)
    print(idx.shape)
    for i in range(n):
        freq[int(idx[i])] += 1
    freq = freq/n
    for i in range(2**m):
        if freq[i] == 0.0: freq[i] = 1e-14
    prob = np.zeros(n)
    for i in range(n):
        prob[i] = freq[int(idx[i])]
    return prob, freq

def probbyfreq(samples, freq):
    (n, m) = samples.shape
    weights = np.array([2**(m-i-1) for i in range(m)])
    idx = to0or1(samples).dot(weights)
    prob = np.zeros(n)
    for i in range(n):
        prob[i] = freq[int(idx[i])]
    return prob



