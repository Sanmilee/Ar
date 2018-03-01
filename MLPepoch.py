
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as ran


#Import training datasets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)




learning_rate = 0.5
epochs = 10
batch_size = 100

total_batch = int(len(mnist.train.labels) / batch_size) + 450   #this equals 1000



#input data
x = tf.placeholder(tf.float32, [None, 784])

#parameters initialization
W = tf.Variable(tf.random_normal([784, 10], stddev=0.03))
b = tf.Variable(tf.random_normal([10]))

#logistic regression computation
y = tf.nn.softmax(tf.matmul(x, W) + b)

#create placeholder for labels
y_ = tf.placeholder(tf.float32, [None, 10])


#compute trainig loss and backpropagate the loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


#create tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)



#epochs and run your model as well as 
for epoch in range(epochs):
	avg_cost = 0
	for i in range(total_batch):
		batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
		c = sess.run(cross_entropy,feed_dict={x: batch_x, y_: batch_y})
		avg_cost += c / total_batch
		sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
	print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))



#evaluate your model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



print("\nModel Accuracy =", (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))




