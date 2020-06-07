import tensorflow as tf

#Attention层
#attention学习博客地址：https://blog.csdn.net/zimiao552147572/article/details/105893842
#attention学习官网地址：http://tensorflow.org/tutorials/text/nmt_with_attention

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_s = tf.keras.layers.Dense(units)
        self.W_h = tf.keras.layers.Dense(units)
        self.W_c = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)


    def __call__(self, dec_hidden, enc_output, enc_pad_mask, use_coverage=False, prev_coverage=None):
        """
         calculate attention and coverage from dec_hidden enc_output and prev_coverage
         one dec_hidden(word) by one dec_hidden
         dec_hidden or query is [batch_sz, enc_unit], enc_output or values is [batch_sz, max_train_x, enc_units],
         prev_coverage is [batch_sz, max_len_x, 1]
         dec_hidden is initialized as enc_hidden, prev_coverage is initialized as None
         output context_vector [batch_sz, enc_units] attention_weights & coverage [batch_sz, max_len_x, 1]
         """
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        if use_coverage and prev_coverage is not None:
            # Multiply coverage vector by w_c to get coverage_features.
            # self.W_s(values) [batch_sz, max_len, units] self.W_h(hidden_with_time_axis) [batch_sz, 1, units]
            # self.W_c(prev_coverage) [batch_sz, max_len, units]  score [batch_sz, max_len, 1]
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis) + self.W_c(prev_coverage)))
            # attention_weights shape (batch_size, max_len, 1)

            # attention_weights sha== (batch_size, max_length, 1)
            mask = tf.cast(enc_pad_mask, dtype=score.dtype)
            masked_score = tf.squeeze(score, axis=-1) * mask
            masked_score = tf.expand_dims(masked_score, axis=2)

            attention_weights = tf.nn.softmax(masked_score, axis=1)

            # attention_weights = masked_attention(enc_pad_mask, attention_weights)
            coverage = attention_weights + prev_coverage
        else:
            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            # 计算注意力权重值
            score = self.V(tf.nn.tanh(
                self.W_s(enc_output) + self.W_h(hidden_with_time_axis)))

            mask = tf.cast(enc_pad_mask, dtype=score.dtype)
            masked_score = tf.squeeze(score, axis=-1) * mask
            masked_score = tf.expand_dims(masked_score, axis=2)
            attention_weights = tf.nn.softmax(masked_score, axis=1)
            # attention_weights = masked_attention(enc_pad_mask, attention_weights)
            if use_coverage:
                coverage = attention_weights
            else:
                coverage = []

        # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights, coverage
