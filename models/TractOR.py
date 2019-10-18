#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model


class TractOR(Model):
    r'''
	TractOR is a model based computing the independent OR over dimensions of the embedding. Its definition is rooted in probabilistic database semantics to allow for tractable querying on low dimensional embeddings.
	'''

    def _calc(self, h, t, r):
        return 1 - tf.reduce_prod(
            1 - h * r * t / (tf.norm(h) * tf.norm(r) * tf.norm(t)),
            -1,
            keep_dims=False)
        # return  - tf.reduce_sum(tf.log(1 - tf.sigmoid(h) * tf.sigmoid(r) * tf.sigmoid(t)), -1, keep_dims = False)

    def embedding_def(self):
        config = self.get_config()
        self.ent_embeddings = tf.get_variable(
                name="ent_embeddings",
                shape=[config.entTotal, config.hidden_size * config.n_mix],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        self.rel_embeddings = tf.get_variable(
                name="rel_embeddings_",
                shape=[config.relTotal, config.hidden_size * config.n_mix],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        self.mix_weights = tf.get_variable(
            name="mix_weights",
            initializer=[1.0 / config.n_mix for i in range(config.n_mix)])
        self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
                                "rel_embeddings":self.rel_embeddings, \
                                      "mix_weights":self.mix_weights}

    def loss_def(self):
        config = self.get_config()
        pos_h, pos_t, pos_r = self.get_positive_instance(in_batch=True)
        neg_h, neg_t, neg_r = self.get_negative_instance(in_batch=True)
        pos_y = self.get_positive_labels(in_batch=True)
        neg_y = self.get_negative_labels(in_batch=True)

        self.loss = 0.0
        p_h = tf.nn.embedding_lookup(self.ent_embeddings, pos_h)
        p_t = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
        p_r = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
        n_h = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
        n_t = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
        n_r = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)
        for i in range(config.n_mix):
                _p_score = self._calc(p_h[:,:,i*config.hidden_size:(i+1)*config.hidden_size], p_t[:,:,i*config.hidden_size:(i+1)*config.hidden_size], p_r[:,:,i*config.hidden_size:(i+1)*config.hidden_size])
                _n_score = self._calc(n_h[:,:,i*config.hidden_size:(i+1)*config.hidden_size], n_t[:,:,i*config.hidden_size:(i+1)*config.hidden_size], n_r[:,:,i*config.hidden_size:(i+1)*config.hidden_size])
                print(_n_score.get_shape())
                loss_func = tf.reduce_mean(
                tf.nn.softplus(-pos_y * _p_score) +
                tf.nn.softplus(-neg_y * _n_score))
                regul_func = tf.reduce_mean(p_h**2 + p_t**2 + p_r**2 + n_h**2 +
                                        n_t**2 + n_r**2)
                self.loss += self.mix_weights[i] * loss_func + config.lmbda * regul_func / config.n_mix

    def predict_def(self):
        config = self.get_config()
        predict_h, predict_t, predict_r = self.get_predict_instance()
        for i in range(config.n_mix):
                predict_h_e = tf.nn.embedding_lookup(self.ent_embeddings[:,i*config.hidden_size:(i+1)*config.hidden_size], predict_h)
                predict_t_e = tf.nn.embedding_lookup(self.ent_embeddings[:,i*config.hidden_size:(i+1)*config.hidden_size], predict_t)
                predict_r_e = tf.nn.embedding_lookup(self.rel_embeddings[:,i*config.hidden_size:(i+1)*config.hidden_size], predict_r)
                if i == 0:
                        self.predict = -self._calc(predict_h_e, predict_t_e, predict_r_e)
                else:
                        self.predict += -self._calc(predict_h_e, predict_t_e, predict_r_e)
