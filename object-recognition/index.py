import gzip, numpy, pickle

f = gzip.open('MNIST_data/mnist.pkl.gz', 'rb')

train_set, valid_set, practice_set = pickle.load(f, encoding='latin1')

print('we gucci')
