#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model


class TractOR(Model):
    r'''
	TractOR is a model based computing the independent OR over dimensions of the embedding. Its definition is rooted in probabilistic database semantics to allow for tractable querying on low dimensional embeddings.
	'''

    def _calc(self, h, t, r):
        print(h.get_shape())
        # return  1 - tf.reduce_prod(1 - tf.math.l2_normalize(h, axis=-1) * tf.math.l2_normalize(t, axis=-1) * tf.math.l2_normalize(t, axis=-1))
        score = 1 - tf.reduce_prod(
            1 - tf.sigmoid(h) * tf.sigmoid(r) * tf.sigmoid(t),
            -1,
            keep_dims=False)
        print(score.get_shape())
        return score

    def embedding_def(self):
        config = self.get_config()
        self.ent_embeddings = tf.get_variable(
            name="ent_embeddings",
            shape=[config.entTotal, config.hidden_size],
            initializer=tf.random_normal_initializer)
        self.rel_embeddings = tf.get_variable(
            name="rel_embeddings",
            shape=[config.relTotal, config.hidden_size],
            initializer=tf.random_normal_initializer)
        self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
              "rel_embeddings":self.rel_embeddings}

    def loss_def(self):
        config = self.get_config()
        pos_h, pos_t, pos_r = self.get_positive_instance(in_batch=True)
        neg_h, neg_t, neg_r = self.get_negative_instance(in_batch=True)
        pos_y = self.get_positive_labels(in_batch=True)
        neg_y = self.get_negative_labels(in_batch=True)

        p_h = tf.nn.embedding_lookup(self.ent_embeddings, pos_h)
        p_t = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
        p_r = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
        n_h = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
        n_t = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
        n_r = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)
        _p_score = self._calc(p_h, p_t, p_r)
        _n_score = self._calc(n_h, n_t, n_r)
        print("here")
        print(_n_score.get_shape())
        loss_func = tf.reduce_mean(
            tf.nn.softplus(-pos_y * _p_score) +
            tf.nn.softplus(-neg_y * _n_score))
        regul_func = tf.reduce_mean(p_h**2 + p_t**2 + p_r**2 + n_h**2 +
                                    n_t**2 + n_r**2)
        self.loss = loss_func + config.lmbda * regul_func

    def predict_def(self):
        config = self.get_config()
        predict_h, predict_t, predict_r = self.get_predict_instance()
        predict_h_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_h)
        predict_t_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_t)
        predict_r_e = tf.nn.embedding_lookup(self.rel_embeddings, predict_r)
        self.predict = -self._calc(predict_h_e, predict_t_e, predict_r_e)
