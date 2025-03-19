import numpy as np


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ------
class CNN:
    def __init__(self):
        self.conv1_filters = np.random.randn(8, 1, 5, 5) * 0.01
        self.conv1_bias = np.zeros((8, 1))
        self.conv2_filters = np.random.randn(16, 8, 5, 5) * 0.01
        self.conv2_bias = np.zeros((16, 1))
        self.fc1_weights = None
        self.fc1_bias = None
        self.fc2_weights = np.random.randn(120, 10) * 0.01
        self.fc2_bias = np.zeros((10, 1))

    def conv_forward(self, x, filters, bias):
        n_filters, d_filter, h_filter, w_filter = filters.shape
        n_x, d_x, h_x, w_x = x.shape
        h_out = h_x - h_filter + 1
        w_out = w_x - w_filter + 1
        
        output = np.zeros((n_x, n_filters, h_out, w_out))
        for i in range(n_x):
            for f in range(n_filters):
                for h in range(h_out):
                    for w in range(w_out):
                        window = x[i, :, h:h+h_filter, w:w+w_filter]
                        output[i, f, h, w] = np.sum(window * filters[f]) + bias[f]
        return output

    def pool_forward(self, x, size=2, stride=2):
        n, d, h, w = x.shape
        h_out = (h - size) // stride + 1
        w_out = (w - size) // stride + 1
        
        output = np.zeros((n, d, h_out, w_out))
        for i in range(n):
            for j in range(d):
                for h in range(h_out):
                    for w in range(w_out):
                        window = x[i, j, h*stride:h*stride+size, w*stride:w*stride+size]
                        output[i, j, h, w] = np.max(window)
        return output

    def flatten(self, x):
        return x.reshape(x.shape[0], -1)

    def fc_forward(self, x, weights, bias):
        return np.dot(x, weights) + bias.T

    def forward(self, x):
        conv1_out = relu(self.conv_forward(x, self.conv1_filters, self.conv1_bias))
        pool1_out = self.pool_forward(conv1_out)
        conv2_out = relu(self.conv_forward(pool1_out, self.conv2_filters, self.conv2_bias))
        pool2_out = self.pool_forward(conv2_out)
        
        flat = self.flatten(pool2_out)
        
        if self.fc1_weights is None:
            self.fc1_weights = np.random.randn(flat.shape[1], 120) * 0.01
            self.fc1_bias = np.zeros((120, 1))
        
        fc1_out = relu(self.fc_forward(flat, self.fc1_weights, self.fc1_bias))
        fc2_out = self.fc_forward(fc1_out, self.fc2_weights, self.fc2_bias)
        return softmax(fc2_out)

    def conv_backward(self, dout, x, filters, bias):
        n_x, d_x, h_x, w_x = x.shape
        n_filters, d_filter, h_filter, w_filter = filters.shape
        h_out = h_x - h_filter + 1
        w_out = w_x - w_filter + 1
        
        dx = np.zeros(x.shape)
        dw = np.zeros(filters.shape)
        db = np.zeros(bias.shape)
        
        for i in range(n_x):
            for f in range(n_filters):
                for h in range(h_out):
                    for w in range(w_out):
                        window = x[i, :, h:h+h_filter, w:w+w_filter]
                        dw[f] += dout[i, f, h, w] * window
                        dx[i, :, h:h+h_filter, w:w+w_filter] += dout[i, f, h, w] * filters[f]
                        db[f] += dout[i, f, h, w]
        return dx, dw, db

    def pool_backward(self, dout, x, size=2, stride=2):
        n, d, h, w = x.shape
        h_out = (h - size) // stride + 1
        w_out = (w - size) // stride + 1
        
        dx = np.zeros(x.shape)
        for i in range(n):
            for j in range(d):
                for h in range(h_out):
                    for w in range(w_out):
                        window = x[i, j, h*stride:h*stride+size, w*stride:w*stride+size]
                        max_idx = np.argmax(window)
                        dx[i, j, h*stride + max_idx//size, w*stride + max_idx%size] = dout[i, j, h, w]
        return dx

    def train(self, x, y, learning_rate=0.01, epochs=10):
        for epoch in range(epochs):
            output = self.forward(x)
            loss = -np.mean(np.log(output[np.arange(len(y)), y]))
            
            dout = output
            dout[np.arange(len(y)), y] -= 1
            dout /= len(y)
            
            dfc2 = np.dot(dout.T, self.fc2_weights.T)
            dw2 = np.dot(self.last_fc1.T, dout)
            db2 = np.sum(dout, axis=0, keepdims=True).T
            
            dfc1 = dfc2 * relu_derivative(self.last_fc1)
            dw1 = np.dot(self.last_flat.T, dfc1)
            db1 = np.sum(dfc1, axis=0, keepdims=True).T
            
            dflat = np.dot(dfc1, self.fc1_weights.T)
            dpool2 = dflat.reshape(self.last_pool2.shape)
            dconv2 = self.pool_backward(dpool2, self.last_conv2)
            dconv2 = dconv2 * relu_derivative(self.last_conv2)
            dpool1, dw2_filters, db2_filters = self.conv_backward(dconv2, self.last_pool1, self.conv2_filters, self.conv2_bias)
            
            dconv1 = self.pool_backward(dpool1, self.last_conv1)
            dconv1 = dconv1 * relu_derivative(self.last_conv1)
            dx, dw1_filters, db1_filters = self.conv_backward(dconv1, x, self.conv1_filters, self.conv1_bias)
            
            self.fc2_weights -= learning_rate * dw2
            self.fc2_bias -= learning_rate * db2
            self.fc1_weights -= learning_rate * dw1
            self.fc1_bias -= learning_rate * db1
            self.conv2_filters -= learning_rate * dw2_filters
            self.conv2_bias -= learning_rate * db2_filters
            self.conv1_filters -= learning_rate * dw1_filters
            self.conv1_bias -= learning_rate * db1_filters
            
            self.last_fc1 = relu(self.fc_forward(self.flatten(self.pool_forward(relu(self.conv_forward(self.pool_forward(relu(self.conv_forward(x, self.conv1_filters, self.conv1_bias))), self.conv2_filters, self.conv2_bias))), self.fc1_weights, self.fc1_bias))
            self.last_flat = self.flatten(self.pool_forward(relu(self.conv_forward(self.pool_forward(relu(self.conv_forward(x, self.conv1_filters, self.conv1_bias))), self.conv2_filters, self.conv2_bias)))
            self.last_pool2 = self.pool_forward(relu(self.conv_forward(self.pool_forward(relu(self.conv_forward(x, self.conv1_filters, self.conv1_bias))), self.conv2_filters, self.conv2_bias)))
            self.last_conv2 = relu(self.conv_forward(self.pool_forward(relu(self.conv_forward(x, self.conv1_filters, self.conv1_bias))), self.conv2_filters, self.conv2_bias))
            self.last_pool1 = self.pool_forward(relu(self.conv_forward(x, self.conv1_filters, self.conv1_bias)))
            self.last_conv1 = relu(self.conv_forward(x, self.conv1_filters, self.conv1_bias))
            
            print(f"Epoch {epoch + 1}, Loss: {loss}")

