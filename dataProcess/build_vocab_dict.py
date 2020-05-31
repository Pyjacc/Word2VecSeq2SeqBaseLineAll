'''
    (1)利用前面清洗后的数据构建vocab词表
        注：vocab是词表,不是用于训练词向量的数据
            后面train_word2vec_model.py中生成的sentences.txt才是语料库,才是用于训练词向量的数据
    (2)保存构建的词表
    (3)注意:是构建词表而不是词向量,词向量要在训练模型后才能得到
'''


from collections import defaultdict
from dataProcess.utils.config import *


# 加载数据的路径
# train_data_x_path = "./dataSetCommon/train_set_seg_x.txt"
# train_data_y_path = "./dataSetCommon/train_set_seg_y.txt"
# test_data_x_path = "./dataSetCommon/test_set_seg_x.txt"
# save_vocab_path = "./dataSetCommon/vocab.txt"
# save_reverse_vocab_path = "./dataSetCommon/reverse_vocab.txt"


def load_data(train_data_x_path, train_data_y_path, test_data_x_path):
    with open(train_data_x_path, mode="r", encoding="utf-8") as f_data_x,\
         open(train_data_y_path, mode="r", encoding="utf-8") as f_data_y,\
         open(test_data_x_path, mode="r", encoding="utf-8") as f_test_data_x:
        words = []
        for line in f_data_x:
            words += line.split(" ")        #第一次也要用“+”,不然只会读取到第二个和第三个文件进行拼接

        for line in f_data_y:
            words += line.split(" ")

        for line in f_test_data_x:
            words += line.split(" ")

    return words


def build_vocab(words_line, sort=True, min_count=0, lower=False):
    '''
    构建词典列表
    :param words: type of list  [word1, word2, word3... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    '''

    result = []
    if sort:
        # sort by count(根据词的频率进行排序)
        # 根据词频进行排序是为了截断（即只选取整个词表中的一部分作为词表）
        dict = defaultdict(int)     # 用于统计词频
        # 取出每一个词,并统计词频
        for words in words_line:
            for word in words.split(" "):
                word = word.strip()
                word = word if not lower else word.lower()
                if not word:
                    continue
                dict[word] += 1     # 统计word出现的次数

        # 根据key进行排序
        dict = sorted(dict.items(), key = lambda d: d[1], reverse=False)
        for index, word in enumerate(dict):
            key = word[0]
            if min_count and min_count > word[1]:   #阈值大于词频,则丢弃该词
                continue
            result.append(key)
    else:
        # sort by items, 即根据词在词表中出现的顺序排序(items即词表中的词)
        for index, word in enumerate(words_line):
            word = word if not lower else word.lower()
            result.append(word)

    vocab_dict = [(word, index) for index, word in enumerate(result)]
    reverse_vocab_dict = [(index, word) for index, word in enumerate(result)]

    return vocab_dict, reverse_vocab_dict



def save_vocab_data(save_data_path, vocab):
    with open(save_data_path, mode="w", encoding="utf-8") as f:
        for line in vocab:
            word, index = line
            f.write("%s\t%d\n" % (word, index))
    f.close()


def save_reverse_vocab_data(save_reverse_vocab_path, reverse_vocab):
    # 不要encoding="utf-8":二进制文件不要制定文件格式
    with open(save_reverse_vocab_path, mode="w")  as f:
        for line in reverse_vocab:
            index, word = line
            f.write("%d\t%s\n" % (index, word))

    f.close()



if __name__ == "__main__":
    # words_line为一行一行的
    words_line = load_data(train_data_x_path, train_data_y_path, test_data_x_path)
    vocab_dict, reverse_vocab_dict = build_vocab(words_line)
    save_vocab_data(save_vocab_path, vocab_dict)
    save_reverse_vocab_data(save_reverse_vocab_path, reverse_vocab_dict)

    print("build vocab success!")

