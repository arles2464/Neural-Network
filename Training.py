# initialization of all variables
initial = tf.global_variables_initializer()

#creating a session
with tf.Session() as sess:
    sess.run(initial)
    writer = tf.summary.FileWriter("/home/tharindra/PycharmProjects/WorkBench/FinalYearProjectBackup/Geetha/TrainResults")
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

    # training loop over the number of epoches
    batchsize=10
    for epoch in range(training_epochs):
        for i in range(len(tr_features)):

            start=i
            end=i+batchsize
            x_batch=tr_features[start:end]
            y_batch=tr_labels[start:end]

            # feeding training data/examples
            sess.run(train_step, feed_dict={X:x_batch , Y:y_batch,keep_prob:0.5})
            i+=batchsize
        # feeding testing data to determine model accuracy
        y_pred = sess.run(tf.argmax(a, 1), feed_dict={X: ts_features,keep_prob:1.0})
        y_true = sess.run(tf.argmax(ts_labels, 1))
        summary, acc = sess.run([merged_summary, accuracy], feed_dict={X: ts_features, Y: ts_labels,keep_prob:1.0})
        # write results to summary file
        writer.add_summary(summary, epoch)
        # print accuracy for each epoch
        print('epoch',epoch, acc)
        print ('---------------')
        print(y_pred, y_true)
