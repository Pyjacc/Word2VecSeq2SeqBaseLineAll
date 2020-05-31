
'''
    (1)从训练集和测试集中读取数据
    (2)重组训练集输入x.训练集输出y,测试集输入x
    (3)对数据进行清洗
    (4)保存清洗后的训练集输入x.训练集输出y,测试集输入x到txt文件
'''


import pandas as pd
import jieba
from jieba import posseg
import multiprocessing as mp
import time
import re
from dataProcess.utils.config import *


# 1. 数据路径
REMOVE_WORDS = ['|', '[', ']', '语音', '图片']      # 自定义要从训练集和测试集中去除的无用词

# train_data_path = "./dataSetCommon/AutoMaster_TrainSet.csv"
# test_data_path = "./dataSetCommon/AutoMaster_TestSet.csv"
# stop_words_path = "./dataSetCommon/哈工大停用词表.txt"

# train_data_x_save_path = "./dataSetCommon/train_set_seg_x.txt"
# train_data_y_save_path = "./dataSetCommon/train_set_seg_y.txt"
# test_data_x_save_path = "./dataSetCommon/test_set_seg_x.txt"



# 2. 读取数据
def parse_train_test_data(train_data_path, test_data_path):
    # 处理训练集数据
    train_data = pd.read_csv(train_data_path, encoding='utf-8')
    # 去除含有无效数据的列
    train_data.dropna(how="any", subset=["Report"], inplace=True)
    # 剩余字段作为输入,包含Brand,Model,Question,Dialogue,如果有空，填充即可
    train_data.fillna("", inplace=True)
    # 取Question和Dialogue两列为训练集x输入
    train_x = train_data.Question.str.cat(train_data.Dialogue)
    train_y = []
    if "Report" in train_data.columns:
        train_y = train_data.Report
        assert len(train_x) == len(train_y)


    # 处理测试集数据
    test_data = pd.read_csv(test_data_path, encoding='utf-8')
    test_data.fillna("", inplace=True)
    test_x = test_data.Question.str.cat(test_data.Dialogue)
    # 测试集没有Report列,因此输出y为空
    test_y = []

    return train_x, train_y, test_x, test_y


# 3. 加载停用词
def load_stop_words(stop_words_path):
    file = open(stop_words_path, mode="r", encoding="utf-8")
    lines = file.readlines()
    stop_words = [stop_word.strip() for stop_word in lines]

    return stop_words


# 4. 切词
'''
    function： 对句子进行切词并返回切分后的词
    input parameters：
    sentence： 要进行切词的句子
    cut_type： 按照什么类型进行切词：
            word： 切割粒度为词（以词为单位进行切词）
            char： 切割粒度为单个字符（以字为单位进行切词）
    pos: 切词后要不要进行词性分析

    output： 返回对句子进行切割后的词（list类型）
'''
def segment(sentence, cut_type="word", pos=False):
    if pos:
        if cut_type == "word":
            word_seq_pos = posseg.lcut(sentence)
            word_seq, word_pos = [], []
            # 第一个位置为词，第二个位置为词性
            for word, p in word_seq_pos:
                word_seq.append(word)
                word_pos.append(p)
            return word_seq, word_pos
        elif cut_type == "char":
            word_seq = list(sentence)
            word_pos = []
            for word in word_seq:
                word_seq_pos = posseg.lcut(word)        # 对单个字符进行切割（获取词和词性）
                word_pos.append(word_seq_pos[0].flag)   # word_seq_pos[0].flag：获取字符对应的词性
            return word_seq, word_pos
    else:
        if cut_type == "word":
            return jieba.lcut(sentence)
        elif cut_type == "char":
            return list(sentence)


 # 5. 去除自定义的无用词
def remove_words(words_list):
    '''
        去除自定义的无用词
    '''
    words = [word for word in words_list if word not in REMOVE_WORDS]
    return words


# 6. 预处理：切词,去除停用词,去除无用词,保存
def preprocess_sentence(data_fram, stop_words, save_data_path):
    with open(save_data_path, mode="w", encoding="utf-8") as f:
        for line in data_fram:
            if isinstance(line, str):
                # 去除特殊字符,用""替换特殊字符
                line = re.sub(r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
                            '', line)
                words_seq, words_pos = segment(line, "word", True)          # 切词
                words_list = remove_words(words_seq)                        # 去除无用词
                word_ist = [word for word in words_list if word not in stop_words]  #去除停用词
                words_line = " ".join(word_ist)
                f.write("%s" % words_line)      # 保存数据到文件
                f.write("\n")
    f.close()           #with打开的文件不用手动close,会自动close


# 7, 保存清洗后的数据
def save_data(data_line, save_data_path):
    with open(save_data_path, mode="w", encoding="utf-8") as f:
        f.write("%s" % data_line)
        f.write("\n")
    f.close()



if __name__ == '__main__':
    # 测试集y为空,不接受
    train_data_x, train_data_y, test_data_x, _ = parse_train_test_data(train_data_path, test_data_path)
    stop_words = load_stop_words(stop_words_path)


    # 单进程运行速度太慢
    # preprocess_sentence(train_data_x, stop_words, train_data_x_save_path)
    # preprocess_sentence(train_data_y, stop_words, train_data_y_save_path)
    # preprocess_sentence(test_data_x, stop_words, test_data_x_save_path)

    # 开启3个子进程,分别处理train_data_x, train_data_y, test_data_x
    time_start = time.time()
    p1 = mp.Process(target=preprocess_sentence, args=(train_data_x, stop_words, train_data_x_save_path))
    p2 = mp.Process(target=preprocess_sentence, args=(train_data_y, stop_words, train_data_y_save_path))
    p3 = mp.Process(target=preprocess_sentence, args=(test_data_x, stop_words, test_data_x_save_path))
    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()

    time_end = time.time()
    print("clean data success!")
    print("total time is : %d seconds" % (time_end - time_start))


