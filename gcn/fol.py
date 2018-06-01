"""

First Order Logic (FOL) rules

"""

import warnings
import numpy as np
import tensorflow as tf


class FOL(object):
    """ First Order Logic (FOL) rules """
    def __init__(self,K, inputs, preds, fea):
        """ Initialize
        : type K: int
        : param K: the number of classes
        """
        self.K = K
        print(self.K)
        self.inputs = inputs
        self.preds = preds
        self.fea = fea
        # Record the data relevance (binary)
        # print(inputs.shape, fea.shape)
        self.conds = self.conditions(self.inputs, self.fea)


    def conditions(self, X, A):
        X = tf.cast(X, tf.float32)
        A = tf.cast(A, tf.float32)
        elems = (X, A)
        conds = tf.map_fn(lambda input: self.condition_single(input), elems=elems, dtype=tf.float32)
        print("shape of conds when initialization", tf.shape(conds))
        return conds

    def distribution_helper(self, w, X, A, conds):
        nx = tf.shape(X)[0]  # see how many instances
        distr = tf.ones([nx, self.K])
        print("shape of conds", conds)
        distr = tf.map_fn(lambda e: self.if_else(e), elems=[conds, X, A, distr], dtype=tf.float32)
        distr = tf.map_fn(lambda d: self.helper(w, d), elems=distr, dtype=tf.float32)
        return distr

    def if_else(self, elem):
        c, x, a, d = elem
        # if does not contains 'but', return distribution (1,1)
        # otherwise, return
        print("x",x)
        print("a",a)
        return tf.cond(tf.equal(c, 1.0), lambda: self.distribution_helper_helper(x, a), lambda: d)

    def distribution_helper_helper(self, x, a):
        # we use range(2) to get 0 and 1, 1 means all parents are in same class
        results = tf.map_fn(lambda k: self.value_single(x, k, a),
                            elems=tf.range(self.K, dtype=tf.int32),
                            dtype=tf.float32)
        return results

    def helper(self, w, d):
        # if not contains 'but', it will become 1 - (1,1) = (0,0)
        # return -w * (tf.reduce_min(d,keepdims=True)-d)
        return -w * (1 - d)


    """
    Interface function of logic constraints

    The interface is general---only need to overload condition_single(.) and
    value_single(.) below to implement a logic rule---but can be slow

    See the overloaded log_distribution(.) of the BUT-rule for an efficient
    version specific to the BUT-rule
    """

    def log_distribution(self, w, X=None, A=None, config={}):
        """ Return an nxK matrix with the (i,c)-th term
        = - w * (1 - r(X_i, y_i=c))    if X_i is a grounding of the rule
        = 1    otherwise
        """
        if A == None:
            X, A, conds = self.inputs, self.fea, self.conds
        else:
            conds = self.conditions(X, A)
        # conds here is same as fea_but_ind which indicate if x contains 'parent'
        log_distr = self.distribution_helper(w, X, A, conds)
        return log_distr





class FOL_Parent(FOL):
    """for first class"""
    def __init__(self, num_class, inputs, preds, fea):
        assert preds.shape[1] == 7
        super(FOL_Parent, self).__init__(num_class, inputs, preds, fea)

    """
    Rule-specific functions
    """
    def condition_single(self, current_input):
        """
        :param current_input: a pair of single instance (x,a)
        :return: 1 if contains 'parent',
                 0 otherwise
        """
        print 'current_input', current_input
        x = current_input[0]
        a = current_input[1]
        return tf.cast(tf.greater_equal(tf.reduce_sum(a), 1.), dtype=tf.float32)


    def value_single(self, x, k, a):
        y = tf.reduce_sum(a * tf.transpose(self.preds), axis=1)
        # print('y',y)
        ret = tf.cond(tf.equal(k,0), lambda: 1., lambda: tf.minimum(1. - y[k], 1.)) # use 0 here, because rule for first class
        # ret = tf.reduce_mean([tf.minimum(1. - y + f[2], 1.), tf.minimum(1. - f[2] + y, 1.)])
        # ret = tf.cast(ret, dtype=tf.float32)
        # input = (x, f)
        # if contains 'but', return ret, otherwise return 1
        # return tf.cast(tf.cond(tf.equal(self.condition_single(input),1.0), lambda: ret, lambda: 1.0),
        #               dtype=tf.float32)
        return tf.cast(ret, dtype=tf.float32)


