import tensorflow as tf
from seq2seq_tf2_d.utils.data_utils import load_word2vec
from seq2seq_tf2_d.encoders import encoder
from seq2seq_tf2_d.attentions import attention
from seq2seq_tf2_d.decoders import decoder

class SequenceToSequence(tf.keras.Model):
    def __init__(self, params):
        super(SequenceToSequence, self).__init__()
        self.embedding_matrix = load_word2vec(params)
        self.params = params
        print(params["batch_size"])
        self.encoder = encoder.Encoder(vocab_size=params["vocab_size"],
                                       embedding_dim=params["embed_size"],
                                       embedding_matrix=self.embedding_matrix,
                                       enc_units=params["enc_units"],
                                       batch_size=params["batch_size"])

        self.attention = attention.BahdanauAttention(units=params["attn_units"])    #attention_units

        self.decoder = decoder.Decoder(vocab_size=params["vocab_size"],
                                       embedding_dim=params["embed_size"],
                                       embedding_matrix=self.embedding_matrix,
                                       dec_units=params["dec_units"],
                                       batch_size=params["batch_size"])


    def call(self, dec_input, dec_hidden, enc_output, dec_target):
        predictions = []
        attentions = []

        context_vector, _ = self.attention(dec_hidden, enc_output)

        #从1开始,是因为起始标志位<START>放在了最前面(ingdex=0),跳过起始标志位
        #dec_target shape:[batch_size, seq_len, embedding_dim]
        for t in range(1, dec_target.shape[1]):
            #训练时,将dec_input作为输入（而不是上一个词的预测值）,即teacher forcing
            pre, dec_hidden = self.decoder(dec_input,
                                           dec_hidden,
                                           enc_output,
                                           context_vector)

            context_vector, attention_weights = self.attention(dec_hidden, enc_output)
            # using teacher forcing
            # 注意时dec_target[:, t],不是dec_target[: t]
            dec_input = tf.expand_dims(dec_target[:, t], 1)   #将真是标签作为输入
            predictions.append(pre)
            attentions.append(attention_weights)

        # tf.stack和tf.concat这两个函数作用类似,都是在某个维度上对矩阵(向量）进行拼接,
        # 不同点在于tf.concat拼接后的矩阵维度不变,tf.stack则会增加一个维度
        return tf.stack(predictions, 1), dec_hidden


    # def call_decoder_onestep(self, dec_input, dec_hidden, enc_output):
    #     # context_vector ()
    #     # attention_weights ()
    #     context_vector, attention_weights = self.attention(dec_hidden, enc_output)
    #
    #     # pred ()
    #     pred, dec_hidden = self.decoder(dec_input,
    #                                     None,
    #                                     None,
    #                                     context_vector)
    #     return pred, dec_hidden, context_vector, attention_weights