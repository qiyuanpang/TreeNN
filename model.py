import tensorflow.compat.v1 as tf
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
    
       
    for i in range(k):
        print(i, 'validation =============================')
        ind1 = list(range(i*m, min(((i+1)*m, n))))
        ind2 = list(set(ind).difference(set(ind1)))
        model.fit(data[ind2, :], labels[ind2], epoches, batch_size)
        loss1 += model.computeloss(data[ind1, :], labels[ind1])
        loss2 += model.computeloss(data[ind2, :], labels[ind2])
             
    print('average    traning loss:', loss2/k)
    print('average validation loss:', loss1/k)
    return loss1/k, loss2/k
        
    

class TreeNN:
    def __init__(self, lr, num_or_size_split, node_hidden_layers, root_hidden_layers, batch_size):
        self.nn_input = self.get_input_layer()
        self.batch_size = batch_size
        splits = tf.split(self.nn_input, num_or_size_split, axis=1)
        #print(len(splits), splits[0].shape)
        cur_top, _ = self.node(splits[0], 2, node_hidden_layers, name='l1_0_')
        for i in range(1, num_or_size_split):
            #print(i, splits[i].shape)
            top, _ = self.node(splits[i], 2, node_hidden_layers, name='l1_' + str(i) + '_')
            cur_top = tf.concat([cur_top, top], axis=1)
        if len(root_hidden_layers) == 0:
            cur_top = tf.reduce_sum(cur_top, 1)
        else:
            cur_top, self.weights = self.node(cur_top, num_or_size_split, root_hidden_layers, name='t_')
        self.output = tf.squeeze(cur_top)
        self.label_placeholder = tf.placeholder("float", [None,], name="label")
        #print(self.output.shape, self.label_placeholder.shape)
        self.custom_loss(reg=0)
        self.model_path = 'model'
        self.deletefiles()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        #loss = self.get_loss_layer()

    def deletefiles(self):
        if os.path.isfile('checkpoint'):
            os.system("rm " + self.model_path + ".index*")
            os.system("rm " + self.model_path + ".meta*")
            os.system("rm " + self.model_path + ".data*")
            os.system("rm checkpoint*")
            
    
    def get_input_layer(self):
        return tf.placeholder("float", [None, None], name="nn_input")
    
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
        return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)

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
                cur_top = tf.nn.sigmoid(tf.matmul(cur_top, weights[layer_step]) + biases[layer_step])
            else:
                cur_top = tf.nn.sigmoid(tf.matmul(cur_top, weights[layer_step]) + biases[layer_step])
        return cur_top, weights


    def fit(self, samples, target, epoches, batch_size, device='/cpu:0'):

        
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
        m = n // self.batch_size
        for i in range(m):
            feed_dict={self.nn_input: samples[i*self.batch_size:(i+1)*self.batch_size, :], self.label_placeholder: labels[i*self.batch_size:(i+1)*self.batch_size]}
            loss += sess.run(self.loss_wo_reg, feed_dict=feed_dict)
        
        return loss/m

    
class FullNN:
    def __init__(self, lr, dim_input, hidden_layers, batch_size):
        self.nn_input = self.get_input_layer()
        cur_top,_ = self.node(self.nn_input, dim_input, hidden_layers, name='f_')
        self.output = tf.squeeze(tf.math.sigmoid(cur_top))
        self.label_placeholder = tf.placeholder("float", [None,], name="label")
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=self.output))
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += 10*sum(reg_loss)
        self.model_path = 'model'
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        #loss = self.get_loss_layer()

    def get_input_layer(self):
        return tf.placeholder("float", [None, None], name="nn_input")

    
    def init_weights(self, shape, name=None):
        return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)

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
                cur_top = tf.nn.tanh(tf.matmul(cur_top, weights[layer_step]) + biases[layer_step])
            else:
                cur_top = tf.nn.tanh(tf.matmul(cur_top, weights[layer_step]) + biases[layer_step])
        return cur_top, weights


    def fit(self, samples, target, epoches, batch_size, device='/cpu:0'):

        
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
            tf.train.Saver().save(sess, self.model_path)

    def predict(self, sample):
        sess = tf.Session()
        tf.train.Saver().restore(sess, self.model_path)
        feed_dict={self.nn_input: sample}
        return sess.run(self.output, feed_dict=feed_dict)
