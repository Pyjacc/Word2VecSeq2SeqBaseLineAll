'''
    构建embedding_matrix
    读取上一步计算的词向量和构建的vocab词表，以vocab中的index为key值构建embedding_matrix
    eg: embedding_matrix[i] = [embedding_vector]
'''


from gensim.models import Word2Vec
import numpy as np
from dataProcess.utils.config import *

# w2v_model_path = "./dataSetCommon/w2v.model"
# save_embedding_matrix_path = "./dataSetCommon/embedding_matrix.txt"


def load_model(w2v_model_path):
    # 注：保存模型的方式不同,加载模型的方式也就不同
    model = Word2Vec.load(w2v_model_path)
    return model


def build_vocab(wv_model):
    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}
    return vocab, reverse_vocab


def get_embeddig_matrix_1(wv_model, save_embedding_path):
    '''
    手动构建embedding matrix
    '''
    # 获取vocab大小
    vocab_size = len(wv_model.wv.vocab)
    # 获取embedding的维度
    embedding_dim = model.wv.vector_size
    print("vocab_size: embedding_dim = %d : %d" % (vocab_size, embedding_dim))

    # 矩阵初始化
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    # 按顺序将词对应的词向量填充进矩阵
    for i in range(vocab_size):
        embedding_matrix[i, :] = wv_model.wv[wv_model.wv.index2word[i]]
    embedding_matrix = embedding_matrix.astype("float32")
    assert embedding_matrix.shape == (vocab_size, embedding_dim)

    np.savetxt(save_embedding_path, embedding_matrix, fmt="%0.8f")
    return embedding_matrix


def get_embedding_matrix_2(wv_model):
    '''
        利用api直接获取embedding matrix
    '''
    embedding_matrix_2  = wv_model.wv.vectors
    return embedding_matrix_2


if __name__ == "__main__":
    model = load_model(w2v_model_path)
    vocab, reverse_vocab = build_vocab(model)
    embedding_matrix = get_embeddig_matrix_1(model, save_embedding_matrix_path)
    print("shape of embedding_matrix :",  embedding_matrix.shape)

    embedding_matrix_2 = get_embedding_matrix_2(model)
    print("shape of embedding_matrix_2 :", embedding_matrix.shape)
    # 判断手动构建的embedding matrix和利用api获取的embedding matrix是否相等
    # print(embedding_matrix == embedding_matrix_2)
