    #cost function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(a),reduction_indices=[1]))
    #optimizer
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
