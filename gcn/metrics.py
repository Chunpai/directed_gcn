import tensorflow as tf
from gcn.utils import *
from fol import *
import scipy.sparse as sp


def masked_softmax_cross_entropy(pi, directed_adj, inputs, preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # print("loss1",loss1)
     # q_y_given_x = tf.nn.softmax(preds)
    # q_preds = tf.sparse_tensor_dense_matmul(directed_adj,preds)
    # tf.stop_gradient(q_preds)


    #TODO Logic Rules
    #TODO get teacher's output
    C = 1.0
    rule_lambda = [1.0]
    q_preds = tf.sparse_tensor_dense_matmul(directed_adj,preds)
    # distr = tf.sparse_tensor_dense_matmul(directed_adj,preds)
    # distr = tf.maximum(tf.minimum(distr,60.),-60.) #truncate to avoid over-/under-flow
    # q_preds = preds * tf.exp(distr)
    # tf.stop_gradient(q_preds)

    # num_class = int(preds.shape[1])
    # p_y_given_x = tf.nn.softmax(preds)
    # rules = [FOL_Parent(num_class, inputs, p_y_given_x, directed_adj)]
    # distr = calc_rule_constraints(rules, rule_lambda, C)
    # print("the shape of distribution",distr)
    # q_preds = preds * distr
    tf.stop_gradient(q_preds)


    loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=q_preds)
    # print("loss2",loss2)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss1 *= mask
    loss2 *= mask
    loss = (1.0-pi)*loss1 + pi*loss2
    return tf.reduce_mean(loss), q_preds


def masked_accuracy(directed_adj, preds, labels, mask):
    #def masked_accuracy(preds, q_preds, labels, mask):
    """Accuracy with masking."""
    p_correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    p_accuracy_all = tf.cast(p_correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    p_accuracy_all *= mask

    q_preds = tf.sparse_tensor_dense_matmul(directed_adj,preds)
    # distr = tf.sparse_tensor_dense_matmul(directed_adj,preds)
    # distr = tf.maximum(tf.minimum(distr,60.),-60.) #truncate to avoid over-/under-flow
    # q_preds = preds * tf.exp(distr)
    q_correct_prediction = tf.equal(tf.argmax(q_preds, 1), tf.argmax(labels, 1))
    q_accuracy_all = tf.cast(q_correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    q_accuracy_all *= mask
    return tf.reduce_mean(p_accuracy_all), tf.reduce_mean(q_accuracy_all)


#TODO calculate the rule constraints
def calc_rule_constraints(rules, rule_lambda, C=1.0):
    distr_all = tf.cast(0, dtype=tf.float32)
    for i,rule in enumerate(rules): #we only use one rule in this application
        print('all data types here \n ')
        distr = rule.log_distribution(C * rule_lambda[i])
        distr_all += distr
    distr_all += distr  #TODO why another addition here, because  distr_all -= distr_y0_copies
    # distr_y0 = distr_all[:,0]
    # distr_y0 = tf.expand_dims(distr_y0,1)
    # distr_y0_copies = tf.tile(distr_y0, [1,distr_all.shape[1]])
    # distr_all -= distr_y0_copies
    distr_all = tf.maximum(tf.minimum(distr_all,60.),-60.) #truncate to avoid over-/under-flow
    return tf.exp(distr_all)



