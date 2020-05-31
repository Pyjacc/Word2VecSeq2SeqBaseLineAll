import tensorflow as tf
from seq2seq_tf2_lj.attentions import attention

#构建decoder层,构建RNN解码器：这里RNN是指GRU, 同时在解码器中使用注意力机制.
#encoder可以用双向gru,但decoder很少用双向gru(若用双向,表示事先已经知道结果了）
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units  # 解码器中gru层中隐藏层神经元数量(通常与embedding_dim大小相等)
        #vocab_size:词表大小
        #embedding_dim：词向量维度
        # embeddig_matrix:为 “词：词向量”形式
        #trainable:是否在Embedding中进行训练
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)  #已经传入了预训练的embedding_matrix,因此不训练
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer="glorot_uniform")
        # Dense操作为线性变换(y = kx + b),类似softmax,但没有做归一化,（Dense后数据仍然是多维的,不是一个值）
        # 可以将Dense替换为tf中的softmax试试看
        self.fc = tf.keras.layers.Dense(vocab_size)     #fully connected layer

        # 在解码器阶段我们将使用注意力机制，这里实例化注意力的类
        self.attention = attention.BahdanauAttention(self.dec_units)


    #prev_dec_hidden, enc_output, context_vector的拼接形式在此处修改
    def call(self, dec_input, prev_dec_hidden, enc_output, context_vector):
        """
        :param dec_input: 每个时间步上解码器的输入
        :param prev_dec_hidden: 前一个节点的的隐层输出
        :param enc_output: 编码器的输出
        :param context_vector: 注意力机制处理结果
        """
        # print("dec_input.shape", dec_input.shape)

        # 输入：(64, 1),64行1列,批量大小句子数为64,1列为该行句子的第N列的单词
        # 输出：(64, 1, 256)即(batch_size, 输入序列最大长度句子的长度, 嵌入维度)
        dec_input = self.embedding(dec_input)       #输入通过embedding层（根据输入词,索引出对应的词向量作为输入）
        # print("dec_input.shape", dec_input.shape)


        #context_vector已经作为参数传入,所以此处不需要再调用attention接口获取context_vector
        # context_vector, attention_weights = self.attention(enc_output, dec_hidden)


        # 将这种'影响程度'与输入dec_input拼接在一起作为本次的GRU网络输入(这个操作也是注意力计算规则的一部分)
        # context_vector shape :(batch_size, embedding_dim)即(64, 1024)
        # tf.expand_dims(context_vector, 1)后 context_vector shape :(batch_size, 1,embedding_dim)即(64, 1, 1024)
        # concat后dec_input shape:(batch_size, 1, embedding_dim + hidden_size),hidden_size即decoder层gru神经元数量
        # concat([(64, 1, 1024),(64, 1, 256)], axis=-1)：1024+256=1280，最终输出 (64, 1, 1280)
        dec_input = tf.concat([tf.expand_dims(context_vector, 1), dec_input], axis=-1)
        # print("dec_input shape", dec_input.shape)
        dec_output, dec_hidden = self.gru(dec_input)  # passing the concatenated vector to the GRU

        # dec_output shape == (batch_size * 1, embedding_dim)
        # 改变输出形状使其适应全连接层的输入形式
        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))

        # pred shape == (batch_size, vocab)
        pred = self.fc(dec_output)  #预测输出
        return pred, dec_hidden


#调用Decoder层
# decoder = Decoder(vocab_size, embedding_dim, embedding_matrix， dec_units, batch_size)
# predice, dnc_output = encoder.call(dec_input, prev_dec_hidden, enc_output, context_vector)
# print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
