import numpy as np
from helper import flip_image

class MultiLayerNeuralNetwork:

    def __init__(self, layer_sizes, C, d, lamb):
        """
        n_hidden_layers: Number of hidden layers
        layer_sizes: sizes of layer (m1,...,mn)
        C: Number of classes
        d: Feature dimension
        """
        self.d = d
        self.C = C
        self.n_hidden_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.lamb = lamb
        self.time = None
        self.initialize()
        
    def set_cyclic_scheme(self, eta_min, eta_max, n_s):
        self.time = 0
        self.cycle = 0
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.n_s = n_s
        def cyclic_update():
            if 2 * self.cycle * self.n_s <= self.time <= (2 * self.cycle + 1) * self.n_s:
                self.eta = self.eta_min + ((self.time - 2 * self.cycle * self.n_s) / self.n_s) * (self.eta_max - self.eta_min)
            elif (2 * self.cycle + 1) * self.n_s <= self.time <= 2 * (self.cycle + 1) * self.n_s:
                self.eta = self.eta_max - ((self.time - (2 * self.cycle + 1) * self.n_s) / self.n_s) * (self.eta_max - self.eta_min)
            self.time += 1
            if self.time >= 2 * (self.cycle + 1) * self.n_s:
                self.cycle += 1
        self.cyclic_update = cyclic_update
    
    def initialize(self, lamb=-1):
        if lamb != -1:
            self.lamb = lamb
        self.W = []
        self.b = []
        if self.n_hidden_layers > 0:
            self.W.append(np.random.normal(0.0, 1/np.sqrt(self.d), (self.layer_sizes[0], self.d)))
            self.b.append(np.zeros((self.layer_sizes[0], 1)))
        for i in range(1, self.n_hidden_layers):
            self.W.append(np.random.normal(0.0, 1/np.sqrt(self.layer_sizes[i - 1]), (self.layer_sizes[i], self.layer_sizes[i - 1])))
            self.b.append(np.zeros((self.layer_sizes[i], 1)))
        if self.n_hidden_layers > 0:
            last_w_dim = (self.C, self.layer_sizes[-1])
            std = 1 / np.sqrt(self.layer_sizes[-1])
        else:
            last_w_dim = (self.C, self.d)
            std = 1 / np.sqrt(self.d)
        self.W.append(np.random.normal(0.0, std, last_w_dim))
        self.b.append(np.zeros((self.C, 1)))

    def __softmax(self, X):
        shift = X - np.max(X)
        return np.exp(shift) / np.sum(np.exp(shift), 0)

    def __compute_gradients_batch(self, X, Y, P, H):
        N = X.shape[1]
        grad_W = []
        grad_b = []
        G = -Y + P
        for i in range(self.n_hidden_layers, 0, -1):
            grad_W.append(np.matmul(G, H[i].T) / N + (2 * self.lamb * self.W[i]))
            grad_b.append(np.matmul(G, np.ones((N, 1))) / N)
            G = np.matmul(self.W[i].T, G)
            G = G * (H[i] > 0)
        grad_W.append(np.matmul(G, X.T) / N + (2 * self.lamb * self.W[0]))
        grad_b.append(np.matmul(G, np.ones((N, 1))) / N)
        grad_W.reverse()
        grad_b.reverse()
        return grad_W, grad_b
    
    def __compute_cost(self, X, Y):
        N = X.shape[1]
        _, P = self.evaluate(X)
        loss = -np.log(P)
        W_sum = 0
        for i in range(self.n_hidden_layers + 1):
            W_sum += np.sum(np.power(self.W[i], 2))
        reg = self.lamb * W_sum
        return np.sum(loss * Y) / N + reg
    
    def __compute_loss(self, X, Y):
        N = X.shape[1]
        _, P = self.evaluate(X)
        loss = -np.log(P)
        return np.sum(loss * Y) / N

    def compute_accuracy(self, X, y):
        _, P = self.evaluate(X)
        arg = np.argmax(P, 0)
        return np.sum(arg.reshape(arg.shape[0], 1) == y) / X.shape[1]
    
    def evaluate(self, X, dropout=False):
        H = [X]
        for i in range(self.n_hidden_layers):
            S = np.matmul(self.W[i], H[i]) + self.b[i]
            h = np.maximum(np.zeros(S.shape), S)
            if dropout and 0 < self.dropout < 1:
                h = (h * (np.random.rand(h.shape[0], h.shape[1]) < self.dropout)) / self.dropout
            H.append(h)
        return H, self.__softmax(np.matmul(self.W[-1], H[-1]) + self.b[-1])
    

    def __mini_batch_gd(self, train_X, train_Y, train_y, val_X, val_Y, val_y, batch_size, augment=False):
        N = train_X.shape[1]
        for i in range(N // batch_size):
            try:
                self.cyclic_update()
                if self.time % (self.n_s // (self.plot_times_per_cycle // 2)) == 0:
                    self.train_cost[self.plot_count] = self.__compute_cost(train_X, train_Y)
                    self.val_cost[self.plot_count] = self.__compute_cost(val_X, val_Y)
                    self.train_loss[self.plot_count] = self.__compute_loss(train_X, train_Y)
                    self.val_loss[self.plot_count] = self.__compute_loss(val_X, val_Y)
                    self.train_acc[self.plot_count] = self.compute_accuracy(train_X, train_y)
                    self.val_acc[self.plot_count] = self.compute_accuracy(val_X, val_y)
                    self.time_steps[self.plot_count] = self.time
                    self.plot_count += 1
            except:
                pass
            start = i * batch_size
            end = (i + 1) * batch_size
            X_batch = train_X[:, start:end]
            Y_batch = train_Y[:, start:end]
            if augment:
                from scipy.ndimage import rotate
                reshaped_batch = X_batch.reshape(3, 32, 32, X_batch.shape[1]).transpose(3, 1, 2, 0)
                # Add photometric jitter
                noise = np.random.normal(0, 0.01, reshaped_batch.shape)
                reshaped_batch += noise
                # Add geometrical jitter aka rotate/flip
                for j in range(X_batch.shape[1]):
                    reshaped_batch[j] = flip_image(reshaped_batch[j])
                    if np.random.rand() > 0.9:
                        reshaped_batch[j] = rotate(reshaped_batch[j], np.random.uniform(-5, 5), reshape=False)
                    
                        
                X_batch = reshaped_batch.transpose(3, 1, 2, 0).reshape(32 * 32 * 3, X_batch.shape[1])
            H, P = self.evaluate(X_batch, dropout=True)
            grad_W, grad_b = self.__compute_gradients_batch(X_batch, Y_batch, P, H)
            for i in range(len(grad_W)):
                self.W[i] -= self.eta * grad_W[i]
                self.b[i] -= self.eta * grad_b[i]

    def fit(self, training_data, validation_data, batch_size=100, eta=0.001, epochs=40, iterator=0, augment=False, dropout=-1):
        self.dropout = dropout
        self.eta = eta
        self.plot_count = 0
        self.plot_times_per_cycle = 10
        if iterator == 0:
            try:
                from tqdm import tqdm
                progress_bar = tqdm(range(epochs))
                print_flag = False
            except: 
                print_flag = True
                progress_bar = range(epochs)
                
        
        train_X = training_data[0]
        train_Y = training_data[1]
        train_y = training_data[2]
        val_X = validation_data[0]
        val_Y = validation_data[1]
        val_y = validation_data[2]
        
        steps = (epochs * (train_X.shape[1] // batch_size)) // ((2 * self.n_s) // self.plot_times_per_cycle)
        self.time_steps = np.zeros(steps)
        self.train_cost = np.zeros(steps)
        self.val_cost = np.zeros(steps)
        self.train_loss = np.zeros(steps)
        self.val_loss = np.zeros(steps)
        self.train_acc = np.zeros(steps)
        self.val_acc = np.zeros(steps)
        
        for epoch in progress_bar:
            if print_flag:
                print("Epoch:", epoch)
            shuffled_indices = np.random.permutation(train_X.shape[1])
            train_X = train_X[:, shuffled_indices]
            train_Y = train_Y[:, shuffled_indices]
            train_y = train_y[shuffled_indices, :]
            progress_bar.set_description("Epoch: " + str(epoch) + ", val_acc: " + str(self.compute_accuracy(val_X, val_y))[:6])
            self.__mini_batch_gd(train_X, train_Y, train_y, val_X, val_Y, val_y, batch_size, augment)

    def __compute_grads_num(self, X, Y, h):
        """ Converted from matlab code """
        from copy import deepcopy
        d = X.shape[0]
        
        b_tmp = deepcopy(self.b)
        W_tmp = deepcopy(self.W)

        grad_W = []
        grad_b = []

        c = self.__compute_cost(X, Y);
        for i in range(len(self.b)):
            gb = np.zeros(self.b[i].shape);
            for j in range(gb.shape[0]):
                self.b[i][j] += h
                c2 = self.__compute_cost(X, Y)
                gb[j] = (c2 - c) / h
                self.b = deepcopy(b_tmp)
            grad_b.append(gb)
            
        for i in range(len(self.W)):
            gW = np.zeros(self.W[i].shape)
            for j in range(gW.shape[0]):
                for k in range(gW.shape[1]):
                    self.W[i][j, k] += h
                    c2 = self.__compute_cost(X, Y)
                    gW[j, k] = (c2 - c) / h
                    self.W = deepcopy(W_tmp)
            grad_W.append(gW)
            
        return grad_W, grad_b

    def check_grad(self, X, Y, h):
        H, P = self.evaluate(X)
        a_W, a_b = self.__compute_gradients_batch(X, Y, P, H)
        n_W, n_b = self.__compute_grads_num(X, Y, h)
        return a_W, a_b, n_W, n_b