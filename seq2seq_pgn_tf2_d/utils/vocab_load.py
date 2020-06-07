
# 字典类
class Vocab:
    '''
    要与dataProcess中train_word2vec_model.py中的class Vocab:类完全相同,否则4个特殊符号及其位置可能对不上
    '''
    def __init__(self, vocab_file, vocab_max_size):
        self.PAD_TOKEN = '<PAD>'        # 如果输入的样本的长度不及阈值，那么剩余的位置补PAD
        self.UNKNOWN_TOKEN = '<UNK>'    # 如果总共的语料库有5万个词，但是我们只选取了3万个词做embedding。当遇到embedding之外的词是，就标注成UNK
        self.START_TOKEN = '<START>'    # 每句话的开头和结尾输入start 和stop， 如果输入样本的长度大于阈值，则截取到阈值，同样输入start和stop
        self.STOP_TOKEN = '<STOP>'

        self.MASK = ['<PAD>', '<UNK>', '<START>', '<STOP>']
        self.MASK_COUNT = len(self.MASK)
        self.pad_token_index = self.MASK.index(self.PAD_TOKEN)
        self.unknown_token_index = self.MASK.index(self.UNKNOWN_TOKEN)
        self.start_token_index = self.MASK.index(self.START_TOKEN)
        self.stop_token_index = self.MASK.index(self.STOP_TOKEN)
        self.word2id, self.id2word = self.load_vocab(vocab_file, vocab_max_size)
        self.count = len(self.word2id)

    def load_vocab(self, vocab_file, vocab_max_size):
        vocab = {mask: index for index, mask in enumerate(self.MASK)}
        reverse_vocab = {index: mask for index, mask in enumerate(self.MASK)}

        for line in open(vocab_file, mode='r', encoding='utf-8').readlines():
            # index, word = line.strip().split('\t')  # 去除每行最后面的\n，然后根据\t分开
            word, index = line.strip().split('\t')
            index = int(index)
            if vocab_max_size and index > vocab_max_size - self.MASK_COUNT - 1:
                break
            vocab[word] = index + self.MASK_COUNT
            reverse_vocab[index + self.MASK_COUNT] = word
        return vocab, reverse_vocab

    # 如果词不在词表中,则用"<UNK>"进行标记
    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[self.UNKNOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self, id):
        if id not in self.id2word:
            # return self.word2id[self.UNKNOWN_TOKEN]
            return self.UNKNOWN_TOKEN
        return self.id2word[id]

    def size(self):
        return self.count