import sys
import os
import tensorflow as tf
import argparse
import pathlib

from seq2seq_pgn_tf2_d.utils import training
from seq2seq_pgn_tf2_d.utils.testing import test_and_save
from seq2seq_pgn_tf2_d.utils.file_utils import get_result_filename
from seq2seq_pgn_tf2_d.utils.config import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# 获取项目工程文件夹所在路径
root = pathlib.Path(os.path.abspath(__file__)).parent.parent
print("BASE_DIR: ", BASE_DIR)    # /home/qinglong/Study/NLP/stage01
print("project root: ", root)    # /home/qinglong/Study/NLP/stage01/chapter4
print("tensorflow version:", tf.__version__)



def gpu_configure():
    # 天坑啊, 在本代码中,如果不注释掉此处的print,就会报如下错误：
    # tensorflow.python.framework.errors_impl.UnknownError: Fail to find the dnn implementation. [Op:CudnnRNN]
    # print("gpu is available: ", tf.test.is_gpu_available())
    print("support gpu: ", tf.test.is_built_with_gpu_support())

    # 设置gpu方式1,设置占用显存自动增长（如果调用GPU失败,则会跑CPU）
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 设置gpu方式2,设置占用显存最高不超过4096(但在本程序中没有生效)
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],
    #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)


def params_set():
    parser = argparse.ArgumentParser()
    # 设置模型参数
    parser.add_argument("--seq2seq_model_dir", default=str(root)+'/seq2seq_pgn_tf2_d/dataSetsPgn/ckpt/seq2seq',
                        help="Save Model folder")
    parser.add_argument("--pgn_model_dir", default=str(root) + '/seq2seq_pgn_tf2_d/dataSetsPgn/ckpt/pgn',
                        help="Save Model folder")
    parser.add_argument("--model_path", default="", type=str,
                        help="Path to a specific model")
    parser.add_argument("--train_seg_x_dir", default=str(root)+'/seq2seq_pgn_tf2_d/dataSetsPgn/train_set_seg_x.txt',
                        help="train_seg_x_dir")
    parser.add_argument("--train_seg_y_dir", default=str(root)+'/seq2seq_pgn_tf2_d/dataSetsPgn/train_set_seg_y.txt',
                        help="train_seg_y_dir")
    parser.add_argument("--test_seg_x_dir", default=str(root)+'/seq2seq_pgn_tf2_d/dataSetsPgn/test_set_seg_x.txt',
                        help="test_seg_x_dir")
    parser.add_argument("--vocab_path", default=str(root)+'/seq2seq_pgn_tf2_d/dataSetsPgn/vocab.txt',
                        help="Vocab path")
    parser.add_argument("--word2vec_output", default=str(root)+'/seq2seq_pgn_tf2_d/dataSetsPgn/word2vec.txt',
                        help="Vocab path")
    parser.add_argument("--test_file_path", default=str(root) + '/seq2seq_pgn_tf2_d/dataSetsPgn/test.csv',
                        help="test_file_path")
    parser.add_argument("--test_result_path", default=str(root) + '/seq2seq_pgn_tf2_d/dataSetsPgn/test_results.csv',
                        help="save test result path")


    parser.add_argument("--max_enc_len", default=200, type=int, help="Encoder input max sequence length")
    parser.add_argument("--max_dec_len", default=50, type=int, help="Decoder input max sequence length")
    parser.add_argument("--max_dec_steps", default=100, type=int, help="maximum number of words of the predicted abstract")
    parser.add_argument("--min_dec_steps", default=30, type=int, help="Minimum number of words of the predicted abstract")
    parser.add_argument("--beam_size", default=3, type=int,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)")
    # 若batch_size设置较大,可能会报OOM错误
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")

    # vocab_size要与构建embedding_matrix时参数vocab_max_size一致,(train_word2vec_model.py中的
    # vocab = Vocab(vocab_path, VOCAB_SIZE)), 否则会出现维度不一致导致错误: Tensor's shape (10000, 256)
    # is not compatible with supplied shape (20000, 256)
    parser.add_argument("--vocab_size", default=30000, type=int, help="Vocabulary size")
    # embed_size要与训练词向量时设置的词向量的维度相同
    parser.add_argument("--embed_size", default=256, type=int, help="Words embeddings dimension")
    parser.add_argument("--enc_units", default=256, type=int, help="Encoder GRU cell units number")
    parser.add_argument("--dec_units", default=256, type=int, help="Decoder GRU cell units number")
    parser.add_argument("--attn_units", default=256, type=int,
                        help="[context vector, decoder state, decoder input] feedforward result dimension - "
                             "this result is used to compute the attention weights")



    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    # 为将代码跑通,可将steps_per_epoch和epochs的值设置较小的值(节省时间)
    parser.add_argument("--steps_per_epoch", default=200, type=int, help="max_train_steps")
    parser.add_argument("--epochs", default=5, type=int, help="train epochs")
    parser.add_argument('--cov_loss_wt', default=1.0, help='Weight of coverage loss (lambda in the paper).'
                            ' If zero, then no incentive to minimize coverage loss.', type=float)

    parser.add_argument("--checkpoints_save_steps", default=10, type=int, help="Save checkpoints every N steps")
    parser.add_argument("--max_steps", default=10000, type=int, help="Max number of iterations")
    parser.add_argument("--num_to_test", default=10, type=int, help="Number of examples to test")



    # set train or test mode here
    parser.add_argument("--mode", default='train', help="training, eval or test options")
    parser.add_argument("--model", default='PGN', help="which model to be slected")
    parser.add_argument("--pointer_gen", default=True, help="training, eval or test options")
    parser.add_argument("--use_coverage", default=True, help="is_coverage")
    parser.add_argument("--greedy_decode", default=False, help="greedy_decoder")
    parser.add_argument("--transformer", default=False, help="transformer")

    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. "
                             "Please refer to the Adagrad optimizer API documentation "
                             "on tensorflow site for more details.",
                        type=float)
    parser.add_argument('--max_grad_norm', default=2.0, help='for gradient clipping', type=float)
    parser.add_argument('--eps', default=1e-12, help='eps', type=float)
    parser.add_argument("--test_save_dir", default=str(root) + '/seq2seq_pgn_tf2_d/dataSetsPgn/', help="test_save_dir")
    # parser.add_argument("--test_df_dir", default=str(root) + '/seq2seq_pgn_tf2_d/dataSetsPgn/test.csv')

    args = parser.parse_args()
    params = vars(args)
    return params


def train_test_seq2seq(params):
    # 训练seq2seq模型
    if params["mode"] == "train":
        print("train start")
        training.train(params)
        print("train success！")

    elif params["mode"] == "test":
        print("test start")
        params['beam_size'] = 2
        params['batch_size'] = 2
        test_and_save(params)
        print("test success")


def main():
    gpu_configure()
    params = params_set()
    train_test_seq2seq(params)


if __name__ == '__main__':
    main()


