import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class TreeNN:
    def __init__(self, lr, num_or_size_split, node_hidden_layers, root_hidden_layers, batch_size):
        self.nn_input = self.get_input_layer()
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
        #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=self.output))
        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.output-self.label_placeholder))
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
