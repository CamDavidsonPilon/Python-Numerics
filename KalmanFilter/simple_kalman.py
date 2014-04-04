#kalman filter, simple example from http://en.wikipedia.org/wiki/Kalman_filter

import numpy as np
from numpy.linalg import inv
from numpy import dot
from matplotlib import pyplot as plt


def predict(x, F, B, u, P, Q ):
    assert x.shape[1] == 1
    assert F.shape[0] == x.shape[0]
    assert u.shape == x.shape
    assert B.shape[0] == u.shape[0]
    assert F.shape[1] == P.shape[0]
    assert Q.shape == P.shape

    x_p = dot(F, x) + dot(B,u)
    P_p = dot(F,P).dot(F.T) + Q

    assert x.shape == x_p.shape
    assert P_p.shape == P.shape

    return x_p, P_p

def update(z, H, x_p, P_p, R ):
    assert H.shape[1] == x_p.shape[0]
    assert H.shape[1] == P_p.shape[0]
    assert R.shape[0] == H.shape[0]
    assert z.shape[1] == 1
    assert z.shape[0] == H.shape[0]

    y = z - dot(H,x_p)
    S = dot(H,P_p).dot(H.T) + R
    K = dot(P_p, H.T).dot(inv(S))
    x_u = x_p + dot(K,y)
    P_u = (np.eye(K.shape[0]) - dot(K,H)).dot(P_p)

    return x_u, P_u


def run(acc_variance=1., obs_variance=1., delta_t = 0.5):
    steps = 100
    X_guesses = np.zeros((2,steps))
    X_actual = np.zeros((2,steps))

    F = np.array([[1, delta_t],[0,1]])
    G = np.array([[delta_t**2/2, delta_t]])
    B = np.zeros((2,2))
    u = np.zeros((2,1))
    Q = np.array([ [delta_t**4/4, delta_t**3/2], [delta_t**3/2, delta_t**2]])*acc_variance
    H = np.array([[1,0]])
    R = np.array( [[obs_variance]])

    #initial values
    x = x_g = np.zeros((2,1))
    P_g = np.zeros((2,2))

    for i in range(steps):
        x = dot(F,x) + np.random.multivariate_normal( [0,0], Q ).reshape( 2, 1)

        x_p, P_p = predict(x_g, F, B, u, P_g, Q )
        z = dot(H,x) + np.random.normal(0,obs_variance)

        x_g, P_g = update(z, H, x_p, P_p, R )

        #print x_g.shape, P_g.shape
        X_guesses[:,i] = x_g[:,0]
        X_actual[:,i] = x[:,0]


    return X_guesses[0,:], X_actual[0,:]


delta_t = 0.5
actual, guesses = run(2.,10., delta_t)

plt.plot(actual, label='actual')
plt.plot(guesses, label='guesses')

plt.legend()
plt.show()





