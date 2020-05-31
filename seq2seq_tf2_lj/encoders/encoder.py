
import tensorflow as tf


#创建encoder层
class Encoder(tf.keras.Model):
    #embedding_matrix:即embedding层,形式为："index:词向量"
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size    #每一次训练时喂给神经网络的样本大小
        self.enc_units = enc_units      # 编码器中gru层中隐藏层神经元数量(通常与词向量维度embedding_dim大小相等)
        self.use_bi_gru = True          #是否使用双向gru

        # 双向gru
        if self.use_bi_gru:
            self.enc_units = self.enc_units // 2

        #实例化embedding层(类似构建了一个字典,训练时输入enc_input,然后索引相应的词向量)
        #weights也可不传入预训练的词向量,采用随机初始化,然后将trainable设置为True
        # vocab_size:词表大小(词表中没有重复词)
        # embedding_dim：词向量维度
        # embeddig_matrix:为 “词：词向量”形式
        # trainable:是否在Embedding中进行训练
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False) #已经传入了预训练的embedding_matrix,因此不训练

        """
        return_sequences：布尔值,是返回输出序列中的最后一个输出还是返回完整序列. 默认值：False.
                True代表返回GRU序列模型的每个时间步的输出(每个输出做连接操作),即返回完整的序列
        return_state：布尔值, 除输出外,是否返回最后一个状态. 默认值：False.
                True代表除了返回输出外,还返回最后一个隐层状态.
        recurrent_initializer：recurrent_kernel权重矩阵的初始化程序,用于对递归状态进行线性转换.默认值：正交.
                'glorot_uniform'即循环状态张量的初始化方式为均匀分布.
        """
        # 实例化gru层
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer="glorot_uniform")
        #双向gru
        self.bi_gru = tf.keras.layers.Bidirectional(self.gru)


    def call(self, enc_input):
        # 对输入进行embedding操作,拿到一次input的所有词的词向量矩阵：(batch_size, seq_len, embedding_dim)
        enc_input_embedded = self.embedding(enc_input)      #根据输入词,索引出对应的词向量
        #初始化（初始化权重等）
        initial_state = self.gru.get_initial_state(enc_input_embedded)

        if self.use_bi_gru:
            # 使用双向GRU
            # initial_state：要传递给单元格的第一个调用的初始状态张量的列表（可选,默认为None,这将导致创建零填充的初始状态张量）
            output, forward_state, backward_state = self.bi_gru(enc_input_embedded,
                                                    initial_state=initial_state * 2)
            enc_hidden = tf.keras.layers.concatenate([forward_state, backward_state], axis=1)
        else:
            #单项GRU
            # 通过gru层获得最后一个时间步的输出和hidden state状态
            output, enc_hidden = self.gru(enc_input_embedded, initial_state=initial_state)

        return output, enc_hidden
