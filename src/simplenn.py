import numpy as np

'''
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
#X = np.array([[0,0,1]])
#y = np.array([[0]]).T
syn0 = 2*np.random.random((3, 4)) - 1
syn1 = 2*np.random.random((4, 1)) - 1
eta = 1
for j in xrange(600000):
    l1 = 1/(1+np.exp(-(np.dot(X, syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1, syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    if (j % 1000) == 0:
        print "Error:" + str(np.mean(np.abs(y-l2)))
    syn1 += eta*l1.T.dot(l2_delta)
    syn0 += eta*X.T.dot(l1_delta)

print 'l2=', l2
'''

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],[1],[1],[0]])
#X = np.array([[1,0,1]])
#y = np.array([[1]])

np.random.seed(1)
eta = 3.0

# randomly initialize our weights with mean 0
#w1 = 2*np.random.random((3,4)) - 1
#w2 = 2*np.random.random((4,1)) - 1
w1 = np.array([[0.1, 0.2, 0.3, 0.4],
               [-0.3, -0.2, -0.1, -0.4],
               [0.3, 0.2, -0.1, -0.2]])
w2 = np.array([[0.3, -0.2, 0.5, -0.4]]).T

for j in xrange(1):

    # Feed forward through layers 0, 1, and 2
    z1 = X.dot(w1)
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)
    a2 = sigmoid(z2)
    print 'a1=', a1
    print 'a2=', a2

    # how much did we miss the target value?
    da2 = a2-y
    print 'da2=', da2

    if (j % 10000) == 0:
        print "Error:" + str(np.mean(np.abs(da2)))

    dz2 = da2 * sigmoid_prime(z2)
    print 'dz2=', dz2

    da1 = dz2.dot(w2.T)
    print 'da1=', da1

    dz1 = da1 * sigmoid_prime(z1)
    print 'dz1=', dz1

    #dx = dz1.dot(w1.T)
    #print 'dx=', dx

    dw2 = a1.T.dot(dz2)
    print 'dw2=', dw2

    dw1 = X.T.dot(dz1)
    print 'dw1=', dw1

    w2 -= eta*dw2/4
    w1 -= eta*dw1/4
