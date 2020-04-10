import numpy as np

def load_batch(filename):
    """ Copied from the dataset website """
    import pickle
    with open(filename, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    y = np.array(dict[b"labels"])
    Y = np.eye(10)[y]
    return np.array(dict[b"data"], dtype=float).T, Y.T, y.reshape((Y.shape[0], 1))

def normalize_cifar(X):
    mean = np.mean(X, 1)
    std = np.std(X, 1)
    return ((X.T - mean) / std).T

def split_training_set(X, Y, y, nr_validation_samples):
    N = X.shape[1] - nr_validation_samples
    val_X = X[:, N:]
    val_Y = Y[:, N:]
    val_y = y[N:, :]
    train_X = X[:, :N]
    train_Y = Y[:, :N]
    train_y = y[:N, :]
    return train_X, train_Y, train_y, val_X, val_Y, val_y


def load_data(number_of_batches, nr_validation_samples):
    dataset = "datasets/cifar-10-batches-py/"
    batches = [dataset + "data_batch_" + str(i + 1)  for i in range(5)] + [dataset + "test_batch"]
    train_X = np.empty((3072, 0))
    train_Y = np.empty((10, 0))
    train_y = np.empty((0, 1))


    for i in range(number_of_batches):
        t_X, t_Y, t_y = load_batch(batches[i])
        train_X = np.hstack([train_X, t_X])
        train_Y = np.hstack([train_Y, t_Y])
        train_y = np.vstack([train_y, t_y.reshape((10000, 1))])

    test_X, test_Y, test_y = load_batch(batches[-1])

    train_X, train_Y, train_y, val_X, val_Y, val_y = split_training_set(train_X, train_Y, train_y, nr_validation_samples)
    
    train_X = normalize_cifar(train_X)
    val_X = normalize_cifar(val_X)
    test_X = normalize_cifar(test_X)
    print("Training: X.shape:", train_X.shape, "Y.shape:", train_Y.shape, "y.shape:", train_y.shape)
    print("Validation: X.shape:", val_X.shape, "Y.shape:", val_Y.shape, "y.shape:", val_y.shape)
    
    return [train_X, train_Y, train_y], [val_X, val_Y, val_y], [test_X, test_Y, test_y]