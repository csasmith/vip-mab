import numpy as np
rng = np.random.default_rng(1)

def KL(p, q): # compute Kullback-Leibler divergence (d in paper). check edge cases.
    if (p == 0 and q == 0) or (p == 1 and q == 1) or p == 0:
        return 0
    elif q == 0 or q == 1:
        return np.inf
    else:
        return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def dKL(p, q): # derivative of KL wrt q. p is constant
    result = (p-q)/(q*(q - 1.0))
    return result

def newton(N, S, k, t, precision = 1e-3, max_iterations = 500, epsilon=1e-12):
    p = S[k]/N[k] # from paper
    q = p + 0.1 # initial guess?
    converged = False

    for n in range(max_iterations):
        f = KL(p, q) - np.log(t)/N[k] # rearrange upper confidence bound eqn
        df = dKL(p, q) # derivative of f is just derivative of KL
        
        if abs(df) < epsilon: # check denominator is not too small
            break
        
        qnew = q - f / df
        if(abs(qnew - q) < precision): # check for early convergence
            converged = True
            print("did converge")
            break
        q = qnew
        
    if(converged == False):
        print("Did not converge")

    return qnew

def KL_UCB(n, K, rwd_means):
    N = np.zeros(K)
    S = np.zeros(K)
    for t in range(K):
        N[t] = 1
        S[t] = rng.uniform(rwd_means[t]-.1, rwd_means[t]+.1) # Use uniform distribution for simplicity
    for t in range(K,n):
        a = np.argmax([newton(N, S, arm, t) for arm in range(K)]) #argmax part of line 6 of algorithm 1
        r = rng.uniform(rwd_means[a]-.1, rwd_means[a]+.1)
        N[a] = N[a] + 1
        S[a] = S[a] + r
    return N,S

K = 5
rwd_means = [.2, .3, .4, .5, .6]
nums, rwds = KL_UCB(1000, 5, rwd_means)
print(nums)
print(rwds)