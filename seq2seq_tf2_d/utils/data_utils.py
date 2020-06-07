
import os
import pickle
import numpy as np


'''
    function: 对数据进行压缩,压缩为二进制格式,节约空间
'''
def dump_pkl(vocab, pkl_path, overwrite=True):
    if not pkl_path:
        print("pkl_path is None")
        return

    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return

    if pkl_path:
        with open(pkl_path, mode="wb") as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(vocab, f, protocol=0)
        print("save %s success!" % pkl_path)


def load_pkl(pkl_path):
    '''
    加载词典文件
    :param pkl_path:
    :return:
    '''
    with open(pkl_path, mode="rb") as f:
        result = pickle.load(f)
    return result


def load_word2vec(params):
    """
    load pretrain word2vec weight matrix
    :param vocab_size:
    :return:
    """

    word2vec_dict = load_pkl(params['word2vec_output'])
    vocab_dict = open(params['vocab_path'], encoding='utf-8').readlines()
    embedding_matrix = np.zeros((params['vocab_size'], params['embed_size']))

    #次部分代码需要简化,因为保存模型中已经是“词:词向量的形式”,因此没必要再重新组装embedding_matrix
    for line in vocab_dict[:params['vocab_size']]:
        word_id = line.split()
        word, i = word_id
        embedding_vector = word2vec_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[int(i)] = embedding_vector

    return embedding_matrix



