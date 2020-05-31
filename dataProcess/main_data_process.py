from dataProcess.clean_data import *
from dataProcess.build_vocab_dict import *
from dataProcess.train_word2vec_model import *


def main():
    # 1. 清洗数据,构建训练集和测试集
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


    # 2. 构建vocab.txt
    time_end = time.time()
    print("clean data success!")
    print("total time is : %d seconds" % (time_end - time_start))

    # words_line为一行一行的
    words_line = load_data(train_data_x_path, train_data_y_path, test_data_x_path)
    vocab_dict, reverse_vocab_dict = build_vocab(words_line)
    save_vocab_data(save_vocab_path, vocab_dict)
    save_reverse_vocab_data(save_reverse_vocab_path, reverse_vocab_dict)

    print("build vocab success!")


    # 3. 训练词向量,获取embedding matrix
    start_time = time.time()
    build_skip_gram_model(train_x_seg_path, train_y_seg_path, test_x_seg_path, w2v_bin_path, sentence_path)
    end_time = time.time()
    print("train model time: %d seconds" % (end_time - start_time))

    load_save_model(w2v_bin_path, vocab_path, save_model_txt_path)
    print("load_save_model success")



if __name__ == "__main__":
    main()

