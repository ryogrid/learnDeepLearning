import tensorflow as tf

# number of inputs
input_len = 2
# number of classes 
classes_num = 2

sess = tf.Session()

x = tf.placeholder("float", [None, input_len])
y_ = tf.placeholder("float", [None, classes_num])

weights1 = tf.Variable(tf.truncated_normal([input_len, 50], stddev=0.0001))
biases1 = tf.Variable(tf.ones([50]))

weights2 = tf.Variable(tf.truncated_normal([50, 25], stddev=0.0001))
biases2 = tf.Variable(tf.ones([25]))

weights3 = tf.Variable(tf.truncated_normal([25, classes_num], stddev=0.0001))
biases3 = tf.Variable(tf.ones([classes_num]))

# This time we introduce a single hidden layer into our model...
hidden_layer_1 = tf.nn.relu(tf.matmul(x, weights1) + biases1)

keep_prob1 = tf.placeholder("float")
hidden_layer_do_1 = tf.nn.dropout(hidden_layer_1, keep_prob1)

hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_do_1, weights2) + biases2)
keep_prob2 = tf.placeholder("float")
hidden_layer_do_2 = tf.nn.dropout(hidden_layer_2, keep_prob2)

model = tf.nn.softmax(tf.matmul(hidden_layer_do_2, weights3) + biases3)

cost = -tf.reduce_sum(y_*tf.log(model))

training_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
                                                                    
init = tf.initialize_all_variables()
sess.run(init)

for ii in range(10000):
    # y_ -> element of correct class only must be 1
    if ii % 2 == 0: # even number [1,2] => 0
        sess.run(training_step, feed_dict={x: [[1, 2]], y_: [[1, 0]], keep_prob1: 1.0, keep_prob2: 1.0})
    else:           # odd number  [2,1] => 1
        sess.run(training_step, feed_dict={x: [[2, 1]], y_: [[0, 1]], keep_prob1: 1.0, keep_prob2: 1.0})

# prediction
print("result of prediction --------")
pred_rslt = sess.run(tf.argmax(model, 1), feed_dict={x: [[1, 2]], keep_prob1: 1.0, keep_prob2: 1.0})
print("  input: [1,2] =>" + str(pred_rslt))
pred_rslt = sess.run(tf.argmax(model, 1), feed_dict={x: [[2, 1]], keep_prob1: 1.0, keep_prob2: 1.0})
print("  input: [2,1] =>" + str(pred_rslt))
