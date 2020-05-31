
# clean_data.py中的文件路径
train_data_path = "./dataSetCommon/AutoMaster_TrainSet.csv"
test_data_path = "./dataSetCommon/AutoMaster_TestSet.csv"
stop_words_path = "./dataSetCommon/哈工大停用词表.txt"

train_data_x_save_path = "./dataSetCommon/train_set_seg_x.txt"
train_data_y_save_path = "./dataSetCommon/train_set_seg_y.txt"
test_data_x_save_path = "./dataSetCommon/test_set_seg_x.txt"


# build_vocab_dict.py中的文件路径
train_data_x_path = "./dataSetCommon/train_set_seg_x.txt"
train_data_y_path = "./dataSetCommon/train_set_seg_y.txt"
test_data_x_path = "./dataSetCommon/test_set_seg_x.txt"
save_vocab_path = "./dataSetCommon/vocab.txt"
save_reverse_vocab_path = "./dataSetCommon/reverse_vocab.txt"


# train_word2vec_model.py中的文件路径
VOCAB_SIZE = 30000
EMBEDDING_DIM = 256
train_x_seg_path = "./dataSetCommon/train_set_seg_x.txt"
train_y_seg_path = "./dataSetCommon/train_set_seg_y.txt"
test_x_seg_path = "./dataSetCommon/test_set_seg_x.txt"
vocab_path = "./dataSetCommon/vocab.txt"         #为调通代码,可只保留vocab.txt中很少的一部分进行调试

sentence_path = './dataSetCommon/sentences.txt'  #用于训练词向量的数据
w2v_bin_path = "./dataSetCommon/w2v.model"       #保存为model或bin文件都行
save_model_txt_path = "./dataSetCommon/word2vec.txt"


# embedding_matrix.py中的文件路径
w2v_model_path = "./dataSetCommon/w2v.model"
save_embedding_matrix_path = "./dataSetCommon/embedding_matrix.txt"


# test_modle.py中的文件路径
w2v_model_path = "./dataSetCommon/w2v_2.model"
sentence_path = './dataSetCommon/sentences.txt'
new_sentence_path = "./dataSetCommon/sentences.txt"

