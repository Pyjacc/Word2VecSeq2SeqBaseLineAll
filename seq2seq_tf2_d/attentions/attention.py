import tensorflow as tf

#Attention层
#attention学习博客地址：https://blog.csdn.net/zimiao552147572/article/details/105893842
#attention学习官网地址：http://tensorflow.org/tutorials/text/nmt_with_attention
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        """初始化三个必要的全连接层:FC:fully connected layers"""
        super(BahdanauAttention, self).__init__()
        #Dense操作即全连接层操作,传入的参数为全连接的那一个层
        self.W1 = tf.keras.layers.Dense(units)  #units: 维数,即隐藏层神经元数量,整数或long
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)       #v是求score的参数,score为单个的值,只有一维
        #注：encoder和decoder的units大小通常一样,attention的units可以不一样


    # 计算score和context vector
    def call(self, dec_hidden, enc_output):
        '''
        输入：
            enc_output:encoder层输出,维度为(batch_size, seq_len, embedding_dim),即
                (batch_size, 输入样本中句子最长的那个样本的长度, 隐藏层中的隐藏神经元数量)
                设batch_size = 64, seq_len = 16, embedding_dim = 1024,则enc_output维度为（64, 16, 1024）
            dec_hidden：解码器的隐层输出状态,维度为(batch_size, embedding_dim),即
                (batch_size, 隐藏层中的隐藏神经元数量),即(64, 1024)
        返回值：
            attention_weights：(batch_size, seq_len, 1)即(64, 16, 1)
            context_vector：(batch_size, embedding_dim)即(64, 1024)
        '''
        # 1. 对dec_hidden进行扩维,dec_hidden原始维度为(batch_size, embedding_dim),扩维后
        # 维度为(batch_size, 1, embedding_dim),即(64, 1, 1024)
        # 增加维度时为了匹配enc_output的维度,进而计算score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        # 2. 计算注意力得分score
        #enc_output：编码器的输出:(64, 16, 1024)。
        #hidden_with_time_axis：解码器的隐层输出状态:(64, 1, 1024)
        #W1和W2：Dense(隐藏层中的隐藏神经元数量1024)
        # tanh(W1(features) + W2(hidden_with_time_axis))：
        # ---> tanh(W1((64, 16, 1024)) + W2((64, 1, 1024)))
        # ---> tanh((64, 16, 1024))
        # ---> (64, 16, 1024) 即(batch_size, seq_len, embedding_dim)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        # 3. 计算注意力权重attention_weights
        #V：Dense(隐藏层中的隐藏神经元数量1)
        # softmax(V(score), axis=1)
        # ---> softmax(V((64, 16, 1024)), axis=1)
        # ---> softmax((64, 16, 1), axis=1)
        # ---> (64, 16, 1) 即(batch_size, seq_len, 1)
        # 因为注意力得分score的形状是(batch_size, seq_len, embedding_dim),
        # 输入(样本)序列最大长度句子的长度(max_seq_len)是输入的长度
        # 因为我们想为每个输入长度分配一个权重,所以softmax应该用作用在第一个轴(max_seq_len)上,因而axis = 1
        # (softmax默认被应用于最后一个轴axis = -1)
        # attention_weights = tf.nn.softmax(self.V(score), axis=1)  #(batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)


        # enc_output (batch_size, enc_len, enc_units)
        # attention_weights (batch_size, enc_len, 1)

        # 4. 计算context vector（获得注意力机制处理后的结果context_vector）
        # 使用注意力权重 * 编码器输出作为返回值,将来会作为解码器的输入
        # reduce_sum(attention_weights * enc_output, axis=1)
        # ---> reduce_sum((64, 16, 1) * (64, 16, 1024), axis=1)
        # ---> reduce_sum((64, 16, 1024), axis=1)
        # ---> (64, 1024) 即(batch_size, embedding_dim)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1) #(batch_size, embedding_dim)

        return context_vector, attention_weights



#调用Attention层
# attention_layer = BahdanauAttention(1024)
# context_vector, attention_weights = attention_layer(enc_output, dec_hidden)
# print("context_vector shape: (batch_size, embedding_dim) {}".format(context_vector.shape)) #(64, 1024)
# print("attention_weights shape: (batch_size, seq_len, 1) {}".format(attention_weights.shape)) #(64, 16, 1)
