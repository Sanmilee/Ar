import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as ran



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)




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





#train the datasets against the labels 1000 iterations

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


#evaluate your model

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #or try “tf.float32”


print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


#check your Weight distribution
# the darker the red the better; white as neutral; and blue as misses.
for i in range(10):
plt.subplot(2, 5, i+1)
weight = sess.run(W)[:,i]
plt.title(i)
plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
frame1 = plt.gca()
frame1.axes.get_xaxis().








