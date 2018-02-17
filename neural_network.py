import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Neural_Network():
    def __init__(self, data, label, file_name):
        self.data = data
        self.label = label
        self.file_name = file_name

        learning_rate = 0.5

        output_neurons = self.data.shape[0]
        input_neurons = self.data.shape[1]
        hidden_neurons_1 = 60

        self.x = tf.placeholder(tf.float32, [None, input_neurons])
        self.y = tf.placeholder(tf.float32, [None, output_neurons])

        hidden_w_1 = tf.Variable(tf.random_normal([input_neurons, hidden_neurons_1], stddev=0.03), name='W1')
        bias_1 = tf.Variable(tf.random_normal([hidden_neurons_1], name='b1'))

        output = tf.Variable(tf.random_normal([hidden_neurons_1, output_neurons], stddev=0.03), name='W2')
        bias_2 = tf.Variable(tf.random_normal([output_neurons], name='b2'))

        hidden_out_1 = tf.add(tf.matmul(self.x, hidden_w_1), bias_1)
        hidden_out_1 = tf.nn.relu(hidden_out_1)

        self.out = tf.nn.softmax(tf.add(tf.matmul(hidden_out_1, output), bias_2))

        y_clipped = tf.clip_by_value(self.out, 1e-10, 0.9999999)
        self.cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(y_clipped) + (1 - self.y) * tf.log(1 - y_clipped), axis=1))

        self.optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cross_entropy)

        self.init_op = tf.global_variables_initializer()

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.out, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.saver = tf.train.Saver({'W1': hidden_w_1, 'b1': bias_1, 'W2': output, 'b2': bias_2})

    def training(self):
        epochs = 1000    #3000
        batch_size = 50  #100

        costs = []
        with tf.Session() as sess:
            sess.run(self.init_op)
            total_batch = int(len(self.label) / batch_size)

            for epoch in range(epochs):
                avg_cost = 0
                for i in range(total_batch):
                    data_batch = self.data[i * batch_size:(i * batch_size + batch_size), ]
                    label_batch = self.label[i * batch_size:(i * batch_size + batch_size), ]
                    _, c = sess.run([self.optimiser, self.cross_entropy],
                                    feed_dict={ self.x: data_batch, self.y: label_batch })
                    avg_cost += c / total_batch
                print("Epoch:", (epoch + 1), "cost=", "{:.3f}".format(avg_cost))
                costs.append(avg_cost)
            print(sess.run(self.accuracy, feed_dict={self.x: self.data, self.y: self.label}))
            print(self.file_name)
            self.saver.save(sess, self.file_name)
        plt.plot(costs)
        plt.show()

    def predict(self, prediction):
        with tf.Session() as sess:
            self.saver.restore(sess, self.file_name)
            label = sess.run(self.out, feed_dict={self.x: prediction})
            item_index = np.where(label == label.max())
            return list(item_index)[1][0]