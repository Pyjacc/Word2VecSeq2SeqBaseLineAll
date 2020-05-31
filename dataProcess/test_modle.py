'''
    加载之前保存的skip-gram模型,测试与输入词相似的词
    新加入语料,在原来模型上进行再次训练
'''


from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time
from dataProcess.utils.config import *a

# w2v_model_path = "./dataSetCommon/w2v_2.model"
# sentence_path = './dataSetCommon/sentences.txt'
# new_sentence_path = "./dataSetCommon/sentences.txt"



def train_word2vec_model(sentence_path, w2v_model_path):
    print("training model!")
    start_time = time.time()
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path), workers=6, size=300, window=5, min_count=1, iter=5)
    w2v.save(w2v_model_path)      # Can be used for continue trainning
    # w2v.wv.save(w2v_bin_path)   # Smaller and faster but can't be trained later
    print("train model time : %d seconds" % (time.time() - start_time))


def load_model(w2v_model_path):
    model = Word2Vec.load(w2v_model_path)
    return model

'''
    添加新的语料, 在原来模型上再次进行训练
'''
def train_model_again(model, sentences_path, w2v_model_path):
    # 获取新的语料
    with open(sentences_path, mode="r", encoding="utf-8") as f:
        data_lines = f.readlines()
    f.close()

    new_words = []
    for line in data_lines:
        word = line.strip().split(" ")
        new_words.append(word)

    # 在原模型上再次进行训练
    model.train(sentences=new_words, epochs=1, total_examples=len(new_words))
    model.save(w2v_model_path)
    return model


if __name__ == "__main__":
    # train_word2vec_model(sentence_path, w2v_model_path)

    # 若保存模型的方式为w2v.wv.save_word2vec_format,则不能直接用Word2Vec.load()方式加载模型,要用
    # KeyedVectors.load_word2vec_format()方式加载
    # model = load_model(w2v_model_path)

    # 再次训练
    # train_model_again(model, new_sentence_path, w2v_model_path)

    model = load_model(w2v_model_path)
    print(model.most_similar("车子"))