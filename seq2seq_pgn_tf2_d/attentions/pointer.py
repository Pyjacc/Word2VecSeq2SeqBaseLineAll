
import tensorflow as tf

class Pointer(tf.keras.layers.Layer):

    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def __call__(self, context_vector, dec_hidden, dec_inp):
        # change dec_inp_context to [batch_sz,embedding_dim+enc_units]
        dec_inp = tf.squeeze(dec_inp, axis=1)
        return tf.nn.sigmoid(self.w_s_reduce(dec_hidden) + self.w_c_reduce(context_vector) + self.w_i_reduce(dec_inp))
