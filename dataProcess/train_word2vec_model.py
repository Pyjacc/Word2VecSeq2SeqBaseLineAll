'''
    生成用于训练词向量的数据sentences.txt
    利用gensim训练skip-gram模型中的词向量,并保存词向量
'''


from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
import time
import numpy as np

from dataProcess.utils.data_utils import dump_pkl
from dataProcess.utils.config import *

# VOCAB_SIZE = 30000
# EMBEDDING_DIM = 256

# # 加载数据的路径
# train_x_seg_path = "./dataSetCommon/train_set_seg_x.txt"
# train_y_seg_path = "./dataSetCommon/train_set_seg_y.txt"
# test_x_seg_path = "./dataSetCommon/test_set_seg_x.txt"
# vocab_path = "./dataSetCommon/vocab.txt"         #为调通代码,可只保留vocab.txt中很少的一部分进行调试
#
# sentence_path = './dataSetCommon/sentences.txt'  #用于训练词向量的数据
# w2v_bin_path = "./dataSetCommon/w2v.model"       #保存为model或bin文件都行
# # w2v_bin_path = "./dataSetCommon/w2v.bin"
# save_model_txt_path = "./dataSetCommon/word2vec.txt"


'''
    function: 返回一行一行的数据
    col_seq:分隔符
'''
def read_lines(path, col_seq=None):
    lines = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if col_seq:
                if col_seq in line:
                    lines.append(line)
            else:
                lines.append(line)

    return lines


'''
    function: 将训练集x数据, 训练集y数据, 测试集x数据组合在一起（为一个很大的句子）
'''
def extract_sentence(train_x_seg_path, train_y_seg_path, test_x_seg_path):
    ret = []
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_x_seg_path)
    for line in lines:
        ret.append(line)
    return ret



def save_sentence(sentence, save_path):
    with open(save_path, mode="w", encoding="utf-8") as f:
        for line in sentence:
            f.write("%s" % line)
    f.close()


'''
    function: 训练词向量, 保存词向量,测试相似性
'''
# 注意,对于样本比较小的数据集,要将min_count的值设置小一点儿,否则会报
def build_skip_gram_model(train_x_seg_path, train_y_seg_path, test_x_seg_path,
                          w2v_bin_path, sentence_path="", min_count=1):
    '''
    使用gensim训练词向量
    :param train_x_seg_path: 训练集x路径
    :param train_y_seg_path: 训练集y路径
    :param test_x_seg_path: 测试集x路径
    :param w2v_bin_path: 保存训练模型的路径
    :param sentence_path: 保存拼接的大句子的路径
    :param min_count: 词频阈值
    :return:
    '''
    sentence = extract_sentence(train_x_seg_path, train_y_seg_path, test_x_seg_path)
    save_sentence(sentence, sentence_path)

    # train skip-gram model的词向量
    print("train w2v model")
    #workers:线程数, size:词向量维度,iter:训练多少轮
    # 使用Word2Vec训练词向量
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path), workers=6, size=EMBEDDING_DIM, window=5, min_count=min_count, iter=5)

    # 使用FastText训练词向量
    # w2v = FastText(sg=1, sentences=LineSentence(sentence_path), workers=8, size=EMBEDDING_DIM, window=5, min_count=min_count, iter=1)

    # 注意：不同的保存方式对应不同的加载模型的方式
    w2v.save(w2v_bin_path)          #可以进行二次训练
    # w2v.wv.save(w2v_bin_path)     #占用存储空间更小,但是不能进行二次训练
    # w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)    #加载时要用KeyedVectors.load_word2vec_format方法加载模型
    print("save %s ok" % w2v_bin_path)


    # test
    sim = w2v.wv.similarity('技师', '车主')
    # sim = w2v.most_similar("技师")
    print('技师 vs 车主 similarity score:', sim)


#加载模型,然后对模型进行压缩保存
def load_save_model(w2v_bin_path, vocab_path, save_txt_path):
    # load model（加载模型的方法）
    # 注意：不同的保存方式对应不同的加载模型的方式
    # skip_gram_model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    skip_gram_model = Word2Vec.load(w2v_bin_path)
    print(skip_gram_model.most_similar("车子"))

    #构建词表：词：词向量
    word_dict = {}

    # 从模型中加载词向量
    # 问题：一次性加载model中的所有词向量（依据词向量构建词表）,如果model中词向量很大，则内存吃不消
    # for word in skip_gram_model.wv.vocab:
    #     word_dict[word] = skip_gram_model[word]     #字典word_dict中的存储形式为： 词:词对应的词向量

    # 构建embedding_matrix
    vocab = Vocab(vocab_path, VOCAB_SIZE)
    for word, index in vocab.word2id.items():
        #注：若要使用腾讯的词向量,只要在加载skip_gram_model时w2v_bin_path用腾讯词向量的路径
        #但是上面vocab_path还是要用自己的vocab.txt文件的路径
        if word in skip_gram_model.wv.vocab:    #构建embedding层
            word_dict[index] = skip_gram_model[word]        # 即为后面所用到的embedding_matrix
        else:
            #随机初始化,值的大小为-0.025到0.025,词向量维度为256
            word_dict[index] = np.random.uniform(-0.025, 0.025, (EMBEDDING_DIM))

    # 将从模型中加载的数据进行压缩保存,保存为二进制文件,节约空间
    dump_pkl(word_dict, save_txt_path, overwrite=True)


#构建词表类
class Vocab:
    def __init__(self, vocab_file_path, vocab_max_size=None):
        '''
        vocab_file_path:词表vocab.txt的路径
        vocab_max_size：从词表中选取多少个词作为新的词表
        '''
        #4个特殊字符
        self.PAD_TOKEN = "<PAD>"
        self.UNKNOWN_TOKEN = "<UNK>"
        self.START_DECODING = "<START>"
        self.END_DECOIDING = "<END>"

        self.MASK = ['<PAD>', '<UNK>', '<START>', '<END>']
        self.MASK_LEN = len(self.MASK)
        self.pad_token_index = self.MASK.index(self.PAD_TOKEN)
        self.unk_token_index = self.MASK.index(self.UNKNOWN_TOKEN)
        self.start_token_index = self.MASK.index(self.START_DECODING)
        self.stop_token_index = self.MASK.index(self.END_DECOIDING)
        self.word2id, self.id2word = self.load_vocab(vocab_file_path, vocab_max_size)
        self.count = len(self.word2id)

    def load_vocab(self, vocab_file_path, vocab_max_size):
        #将4个特殊字符放在新构建的词表的最前面
        #构建词表,格式为{词：索引}
        vocab = {mask:index for index, mask in enumerate(self.MASK)}
        reverse_vocab = {index:mask for index, mask in enumerate(self.MASK)}

        # 从原有词表vocab.txt中截取一部分词加入到新构建的词表中
        for line in open(vocab_file_path, "r", encoding="utf-8").readlines():
            word, index = line.strip().split("\t")  #原词表vocab.txt中使用“\t”分割词和index
            index = int(index)
            if vocab_max_size and index > vocab_max_size - self.MASK_LEN - 1:
                break
            vocab[word] = index + self.MASK_LEN         #将加载的词放到4个特殊字符后面
            reverse_vocab[index+self.MASK_LEN] = word
        return vocab, reverse_vocab


    #如果词不在词表中,则用"<UNK>"进行标记
    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[self.UNKNOWN_TOKEN]
        return self.word2id[word]


    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            return self.id2word[self.unk_token_index]
            # return self.UNKNOWN_TOKEN
        return self.id2word[word_id]

    def size(self):
        return self.count



if __name__ == "__main__":
    start_time = time.time()
    build_skip_gram_model(train_x_seg_path, train_y_seg_path, test_x_seg_path, w2v_bin_path, sentence_path)
    end_time = time.time()
    print("train model time: %d seconds" % (end_time - start_time))

    load_save_model(w2v_bin_path, vocab_path, save_model_txt_path)
    print("load_save_model success")
