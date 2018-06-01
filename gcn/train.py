from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from scipy import sparse
from gcn.models import GCN, MLP

# Set random seed
# seed = 3
# np.random.seed(seed)
# tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

all_p_acc = []
all_q_acc = []
for run in range(10):
    print("-----------------Running on #{} of {} dataset------------------".format(run, FLAGS.dataset))
    total_p_acc = []
    total_q_acc = []
    # Load data
    adj, adj_pa, fea, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset, run)
    # adj is sparse matrix
    # print("train_mask",len(train_mask))
    # print("true mast", len([e for e in train_mask if e == True]))
    #TODO add self-loop in adj, self-loop may has different value rather than 1
    dense_adj = np.array(adj.todense())
    dense_adj_pa = np.array(adj_pa.todense())
    dense_adj_ch = np.array(dense_adj - dense_adj_pa)
    directed_adj = adj_pa.todense()
    size = dense_adj_pa.shape[0]
    for i in range(size):
        directed_adj[i,i] = 1
    sparse_adj = sparse.csr_matrix(directed_adj)
    directed_adj = sparse_to_tuple(sparse_adj)
    # directed_adj = sparse_to_tuple(adj_pa)

    # Some preprocessing
    dense_features = np.array(fea.todense())
    features = preprocess_features(fea)

    for seed in range(110,120):
        # Set random seed
        np.random.seed(seed)
        tf.set_random_seed(seed)

        if FLAGS.model == 'gcn':
            support = [preprocess_adj(adj)]
            num_supports = 1
            model_func = GCN
        elif FLAGS.model == 'gcn_cheby':
            support = chebyshev_polynomials(adj, FLAGS.max_degree)
            num_supports = 1 + FLAGS.max_degree
            model_func = GCN
        elif FLAGS.model == 'dense':
            support = [preprocess_adj(adj)]  # Not used
            num_supports = 1
            model_func = MLP
        else:
            raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

        # Define placeholders
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            'dense_features': tf.placeholder(tf.float32),
            # 'directed_adj': tf.placeholder(tf.float32),
            'directed_adj': tf.sparse_placeholder(tf.float32),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }

        # Create model
        model = model_func(placeholders, input_dim=features[2][1], logging=True)

        session_conf = tf.ConfigProto(allow_soft_placement=True)

        # Initialize session
        sess = tf.Session(config=session_conf)


        # Define model evaluation function
        def evaluate(directed_adj, features, dense_features , support, labels, mask, placeholders):
            t_test = time.time()
            feed_dict_val = construct_feed_dict(directed_adj,
                                                features,
                                                dense_features,
                                                support, labels,
                                                mask,
                                                placeholders)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            return outs_val[0], outs_val[1], (time.time() - t_test)


        def get_pi(cur_iter, params=None, pi=None):
            """exponential decay: pi_t = max{1-k^t,lb}"""
            k,lb = params[0], params[1]
            pi = 1.0 - max([k**cur_iter,lb])
            return pi
        # Init variables
        sess.run(tf.global_variables_initializer())

        cost_val = []



        # Train model
        for epoch in range(FLAGS.epochs):
            t = time.time()
            pi = get_pi(cur_iter=epoch*1.0/FLAGS.epochs, params=[0.2,0.0])
            # pi = 0.5
            model.set_pi(pi)
            # Construct feed dictionary
            feed_dict = construct_feed_dict(directed_adj,
                                            features,
                                            dense_features,
                                            support,
                                            y_train,
                                            train_mask,
                                            placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            #TODO add logic rule in the model.
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

            # Validation
            cost, acc, duration = evaluate(directed_adj, features,
                                           dense_features, support,
                                           y_val, val_mask, placeholders)
            cost_val.append(cost)

            # Print results
            # print("Epoch:", '%04d' % (epoch + 1),
            #       "pi=", "{:.5f}".format(pi),
            #       "train_loss=", "{:.5f}".format(outs[1]),
            #       "[p] train_acc=", "{:.5f}".format(outs[2][0]),
            #       "[q] train_acc=", "{:.5f}".format(outs[2][1]),
            #       "val_loss=", "{:.5f}".format(cost),
            #       "[p] val_acc=", "{:.5f}".format(acc[0]),
            #       "[q] val_acc=", "{:.5f}".format(acc[1]),
            #       "time=", "{:.5f}".format(time.time() - t))

            # test_cost, test_acc, test_duration = evaluate(directed_adj, features, support, y_test, test_mask, placeholders)
            # print("Test set results:", "cost=", "{:.5f}".format(test_cost),
            #       "[p] accuracy=", "{:.5f}".format(test_acc[0]),
            #       "[q] accuracy=", "{:.5f}".format(test_acc[1]),
            #       "time=", "{:.5f}".format(test_duration))

            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                print("Early stopping...")
                break

        # print("Optimization Finished!")

        # Testing
        test_cost, test_acc, test_duration = evaluate(directed_adj, features,
                                                      dense_features, support,
                                                      y_test, test_mask, placeholders)
        all_p_acc.append(test_acc[0])
        all_q_acc.append(test_acc[1])
        total_p_acc.append(test_acc[0])
        total_q_acc.append(test_acc[1])
        print("[#{}] Test set results:".format(seed), "cost=", "{:.5f}".format(test_cost),
              "[p] accuracy=", "{:.5f}".format(test_acc[0]),
              "[q] accuracy=", "{:.5f}".format(test_acc[1]),
              "time=", "{:.5f}".format(test_duration))
    print("Run #{} Average Test Accuracy [p]: {:.5f}, [q]: {:.5f} \n".format(run,np.mean(total_p_acc), np.mean(total_q_acc)))

print("Average Test Accuracy [p]: {:.5f}, [q]: {:.5f} \n".format(np.mean(all_p_acc), np.mean(all_q_acc)))
