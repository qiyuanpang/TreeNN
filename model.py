import tensorflow.compat.v1 as tf
from tensorflow.keras import layers, models
tf.disable_v2_behavior()
import numpy as np
import os

def kfoldvalidation(model, data, labels, epoches = 10, batch_size = 24, k=5):
    n = len(data)
    assert n == len(labels)
    m = int(np.ceil(1.0*n/k))
    ind = list(range(n))
    loss1 = 0.0
    loss2 = 0.0
    
    save = True
    for i in range(k):
        print(i, 'validation =============================')
        ind1 = list(range(i*m, min(((i+1)*m, n))))
        ind2 = list(set(ind).difference(set(ind1)))
        #if i == k-1: save = True
        model.fit(data[ind2, :], labels[ind2], epoches, batch_size, save)
        loss1 += model.computeloss(data[ind1, :], labels[ind1])
        loss2 += model.computeloss(data[ind2, :], labels[ind2])
             
    print('average    traning loss:', loss2/k)
    print('average validation loss:', loss1/k)
    return loss1/k, loss2/k

def rekfoldvalidation(model, data, labels, epoches = 10, batch_size = 24, k=5):
    n = len(data)
    assert n == len(labels)
    m = int(np.ceil(1.0*n/k))
    ind = list(range(n))
    loss1 = 0.0
    loss2 = 0.0
    
    save = True
    for i in range(k):
        print(i, 'validation =============================')
        ind1 = list(range(i*m, min(((i+1)*m, n))))
        ind2 = list(set(ind).difference(set(ind1)))
        #if i == k-1: save = True
        model.refit(data[ind2, :], labels[ind2], epoches, batch_size, save)
        loss1 += model.computeloss(data[ind1, :], labels[ind1])
        loss2 += model.computeloss(data[ind2, :], labels[ind2])
             
    print('average    traning loss:', loss2/k)
    print('average validation loss:', loss1/k)
    return loss1/k, loss2/k
        
    

class NNModel:        
    def __init__(self, modelname, params):
        self.params = params
        self.modelname = modelname
        if self.modelname == 'TreeNN':
            self.TreeNN()
        elif self.modelname == 'DenseNN':
            self.DenseNN()
        elif self.modelname == 'CNN':
            self.CNN()
        else:
            assert 1 == 0, 'You have to specify a model name!'
        self.model_path = 'nnmodel'
        #self.deletefiles()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.params['lr']).minimize(self.loss)
        #self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.params['lr']).minimize(self.loss)
        #self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.params['lr'], momentum=0.9).minimize(self.loss)
        #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.params['lr']).minimize(self.loss)

    def CNN(self):
        self.nn_input = self.get_input_layer(self.params['input_shape'])
        output = tf.keras.layers.Conv2D(self.params['filters'], self.params['kernel_size'], activation=self.params['activation'], padding='same', input_shape=self.params['input_shape'][1:])(self.nn_input)
        for i in range(1, self.params['num_layers']):
            output = tf.keras.layers.Conv2D(self.params['filters'], self.params['kernel_size'], activation=self.params['activation'], padding='same', input_shape=(output.shape[1], output.shape[2], self.params['filters']))(output)
        print(output.shape)
        output = tf.keras.layers.Flatten()(output)
        print(output.shape)
        output = tf.keras.layers.Dense(self.params['input_shape'][1], activation=self.params['activation'])(output)
        self.output = tf.squeeze(tf.keras.layers.Dense(1)(output))
        print(self.output.shape)
        self.label_placeholder = tf.placeholder("float", [None,], name="label")
        #print(self.output.shape, self.label_placeholder.shape)
        self.custom_loss(reg=0)

    def DenseNN(self):
        self.nn_input = self.get_input_layer()
        self.batch_size = self.params['batch_size']
        cur_top,_ = self.lastnode(self.nn_input, self.params['dim_input'], self.params['hidden_layers'], name='f_')
        self.output = tf.squeeze(tf.math.sigmoid(cur_top))
        self.label_placeholder = tf.placeholder("float", [None,], name="label")
        self.custom_loss(reg=0)

    def TreeNN(self):
        self.nn_input = self.get_input_layer()
        self.batch_size = self.params['batch_size']
        splits = tf.split(self.nn_input, self.params['num_or_size_split'], axis=1)
        #print(len(splits), splits[0].shape)
        cur_top, _ = self.node(splits[0], self.params['num_or_size_split'][0], self.params['node_hidden_layers'], name='l1_0_')
        for i in range(1, len(self.params['num_or_size_split'])):
            #print(i, splits[i].shape)
            top, _ = self.node(splits[i], self.params['num_or_size_split'][i], self.params['node_hidden_layers'], name='l1_' + str(i) + '_')
            cur_top = tf.concat([cur_top, top], axis=1)
        if len(self.params['root_hidden_layers']) == 0:
            cur_top = tf.reduce_sum(cur_top, 1)
        else:
            cur_top, self.weights = self.lastnode(cur_top, len(self.params['num_or_size_split']), self.params['root_hidden_layers'], name='t_')
        self.output = tf.squeeze(cur_top)
        print(self.output.shape)
        self.label_placeholder = tf.placeholder("float", [None,], name="label")
        #print(self.output.shape, self.label_placeholder.shape)
        self.custom_loss(reg=0)
        
        #loss = self.get_loss_layer()
   
    def savemodel(self):
        with tf.Session() as sess:
             tf.train.Saver().save(sess, self.model_path) 
   
    def deletefiles(self):
        if os.path.isfile('checkpoint'):
            os.system("rm " + self.model_path + ".index*")
            os.system("rm " + self.model_path + ".meta*")
            os.system("rm " + self.model_path + ".data*")
            os.system("rm checkpoint*")
            
    
    def get_input_layer(self, shape=None):
        if shape is None:
            return tf.placeholder("float", [None, None], name="nn_input")
        else:
            return tf.placeholder("float", shape, name="nn_input")
    
    def custom_loss(self, reg=0):
        #loss = tf.reduce_mean(tf.nn.l2_loss(self.output-self.label_placeholder))
        #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_placeholder, logits=self.output))
        #loss = tf.square(tf.reduce_mean(-tf.reduce_sum(self.output*tf.log(self.label_placeholder))+tf.reduce_sum(self.output*tf.log(self.output))))
        loss = tf.reduce_mean(-tf.reduce_sum(self.output*tf.log((self.label_placeholder+self.output)*0.5))+tf.reduce_sum(self.output*tf.log(self.output)))*0.5 + tf.reduce_mean(-tf.reduce_sum(self.label_placeholder*tf.log((self.label_placeholder+self.output)*0.5))+tf.reduce_sum(self.label_placeholder*tf.log(self.label_placeholder)))*0.5
        #loss += 1.1*tf.square(tf.reduce_mean((tf.reduce_sum(self.output)-tf.reduce_sum(self.label_placeholder))))
        #loss += tf.reduce_mean(tf.nn.l2_loss(self.output-self.label_placeholder))
        self.loss_wo_reg = loss
        if reg > 0:
            loss += reg*sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = loss
    
    def init_weights(self, shape, name=None):
        return tf.Variable(tf.random_normal(shape, mean=0.2, stddev=0.1), name=name)

    def init_bias(self, shape, name=None):
        return tf.Variable(tf.zeros(shape), name=name)

    def node(self, nn_input, dim_input, hidden_layers, name=''):
        dim_hidden = hidden_layers + [1]
        n_layers = len(dim_hidden)
        weights = []
        biases = []
        in_shape = dim_input
        for layer_step in range(0, n_layers):
            cur_weight = self.init_weights([in_shape, dim_hidden[layer_step]], name=name+'w_' + str(layer_step))
            cur_bias = self.init_bias([dim_hidden[layer_step]], name=name+'b_' + str(layer_step))
            in_shape = dim_hidden[layer_step]
            weights.append(cur_weight)
            biases.append(cur_bias)
        cur_top = nn_input
        if len(cur_top.shape) == 1: cur_top = tf.expand_dims(cur_top, 0)
        #print(cur_top.shape, '???')
        for layer_step in range(0, n_layers):
            #print(layer_step, cur_top.shape)
            if layer_step != n_layers-1:  # final layer has no RELU
                cur_top = tf.matmul(cur_top, weights[layer_step]) + biases[layer_step]
            else:
                cur_top = tf.matmul(cur_top, weights[layer_step]) + biases[layer_step]
        return cur_top, weights

    def lastnode(self, nn_input, dim_input, hidden_layers, name=''):
        dim_hidden = hidden_layers + [1]
        n_layers = len(dim_hidden)
        weights = []
        biases = []
        in_shape = dim_input
        for layer_step in range(0, n_layers):
            cur_weight = self.init_weights([in_shape, dim_hidden[layer_step]], name=name+'w_' + str(layer_step))
            cur_bias = self.init_bias([dim_hidden[layer_step]], name=name+'b_' + str(layer_step))
            in_shape = dim_hidden[layer_step]
            weights.append(cur_weight)
            biases.append(cur_bias)
        cur_top = nn_input
        if len(cur_top.shape) == 1: cur_top = tf.expand_dims(cur_top, 0)
        #print(cur_top.shape, '???')
        for layer_step in range(0, n_layers):
            #print(layer_step, cur_top.shape)
            if layer_step != n_layers-1:  # final layer has no RELU
                cur_top = tf.nn.sigmoid(tf.matmul(cur_top, weights[layer_step]) + biases[layer_step])
            else:
                cur_top = tf.matmul(cur_top, weights[layer_step]) + biases[layer_step]
        return cur_top, weights


    def fit(self, samples, target, epoches, batch_size, save=False):

        
        idx = np.array(range(len(samples)))
        np.random.shuffle(idx)
        batches = int(len(samples)/batch_size)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            loss = 0.0
            
            for batch in range(batches):
                feed_dict = {self.nn_input: samples[idx[batch*batch_size: (batch+1)*batch_size], :], self.label_placeholder: target[idx[batch*batch_size: (batch+1)*batch_size]]}
                #print(feed_dict)
                loss += sess.run(self.loss, feed_dict=feed_dict)
            print("initial loss:", loss/batches)
            
            for epoch in range(epoches):
                idx1 = np.array(range(len(samples)))
                np.random.shuffle(idx1)
                #print(sess.run(self.weights[2]))
                for batch in range(batches):
                    feed_dict = {self.nn_input: samples[idx1[batch*batch_size: (batch+1)*batch_size], :], self.label_placeholder: target[idx1[batch*batch_size: (batch+1)*batch_size]]}
                    sess.run(self.optimizer, feed_dict=feed_dict)
                loss = 0.0
                for batch in range(batches):
                    feed_dict = {self.nn_input: samples[idx[batch*batch_size: (batch+1)*batch_size], :], self.label_placeholder: target[idx[batch*batch_size: (batch+1)*batch_size]]}
                    loss += sess.run(self.loss, feed_dict=feed_dict)
                print("loss after epoch", epoch, ":", loss/batches)
            if save:
                self.deletefiles()
                tf.train.Saver().save(sess, self.model_path)

    def predict(self, sample):
        sess = tf.Session()
        tf.train.Saver().restore(sess, self.model_path)
        feed_dict={self.nn_input: sample}
        return sess.run(self.output, feed_dict=feed_dict)
        
    def computeloss(self, samples, labels):
        sess = tf.Session()
        tf.train.Saver().restore(sess, self.model_path)
        loss = 0.0
        n = len(samples)
        m = n // self.params['batch_size']
        for i in range(max(1, m)):
            feed_dict={self.nn_input: samples[i*self.params['batch_size']:min((i+1)*self.params['batch_size'], n), :], self.label_placeholder: labels[i*self.params['batch_size']:min((i+1)*self.params['batch_size'], n)]}
            loss += sess.run(self.loss_wo_reg, feed_dict=feed_dict)
        
        if m > 0: return loss/m
        else: return loss

    def refit(self, samples, target, epoches, batch_size, save):

        
        idx = np.array(range(len(samples)))
        np.random.shuffle(idx)
        batches = int(len(samples)/batch_size)
        with tf.Session() as sess:
            #init_op = tf.global_variables_initializer()
            #sess.run(init_op)
            tf.train.Saver().restore(sess, self.model_path)
            loss = 0.0
            
            for batch in range(batches):
                feed_dict = {self.nn_input: samples[idx[batch*batch_size: (batch+1)*batch_size], :], self.label_placeholder: target[idx[batch*batch_size: (batch+1)*batch_size]]}
                #print(feed_dict)
                loss += sess.run(self.loss, feed_dict=feed_dict)
            print("initial loss:", loss/batches)
            
            for epoch in range(epoches):
                idx1 = np.array(range(len(samples)))
                np.random.shuffle(idx1)
                #print(sess.run(self.weights[2]))
                for batch in range(batches):
                    feed_dict = {self.nn_input: samples[idx1[batch*batch_size: (batch+1)*batch_size], :], self.label_placeholder: target[idx1[batch*batch_size: (batch+1)*batch_size]]}
                    sess.run(self.optimizer, feed_dict=feed_dict)
                loss = 0.0
                for batch in range(batches):
                    feed_dict = {self.nn_input: samples[idx[batch*batch_size: (batch+1)*batch_size], :], self.label_placeholder: target[idx[batch*batch_size: (batch+1)*batch_size]]}
                    loss += sess.run(self.loss, feed_dict=feed_dict)
                print("loss after epoch", epoch, ":", loss/batches)
            if save:
                self.deletefiles()
                tf.train.Saver().save(sess, self.model_path)
