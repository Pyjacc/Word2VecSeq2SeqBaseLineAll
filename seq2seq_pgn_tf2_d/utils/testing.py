# -*- coding:utf-8 -*-

import tensorflow as tf


from seq2seq_pgn_tf2_d.utils.config import checkpoint_dir
from seq2seq_pgn_tf2_d.utils.batcher_utils import batcher
from seq2seq_pgn_tf2_d.models.pgn import PGN
from seq2seq_pgn_tf2_d.utils.vocab_load import Vocab
from seq2seq_pgn_tf2_d.utils.gpu_utils import config_gpu
from seq2seq_pgn_tf2_d.utils.test_helper import beam_decode
# from seq2seq_pgn_tf2_d.utils.testing import save_predict_result


#
# def test(params):
#     assert params["mode"].lower() in ["test", "eval"], "change training mode to 'test' or 'eval'"
#     assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
#     # GPU资源配置
#     # config_gpu(use_cpu=True)
#     gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#     if gpus:
#         tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
#         tf.config.experimental.set_memory_growth(gpus[0], enable=True)
#
#     print("Building the model ...")
#     model = PGN(params)
#
#     print("Creating the vocab ...")
#     vocab = Vocab(params["vocab_path"], params["vocab_size"])
#     params['vocab_size'] = vocab.count
#
#     print("Creating the batcher ...")
#     b = batcher(vocab, params)
#
#     print("Creating the checkpoint manager")
#     checkpoint = tf.train.Checkpoint(PGN=model)
#     checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
#     checkpoint.restore(checkpoint_manager.latest_checkpoint)
#     if checkpoint_manager.latest_checkpoint:
#         print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
#     else:
#         print("Initializing from scratch.")
#     print("Model restored")
#     result = []
#
#     if params['mode'] == 'test':
#         for batch in b:
#             # yield beam_decode(model, batch, vocab, params)
#             result.append(beam_decode(model, batch, vocab, params))
#     return result
#
#
# def predict_result(params, result_save_path):
#     # 预测结果
#     results = test(params)
#     print('prediction results is : ',results)
#     # 保存结果
#     # save_predict_result(results, result_save_path)


# if __name__ == '__main__':
#     # 获得参数
#     params = get_params()
#     params['batch_size'] = 3
#     params['beam_size'] = 3
#     params['mode'] = 'test'
#     test(params)



import tensorflow as tf
import pandas as pd


def test(params):
    global model, ckpt, checkpoint_dir
    assert params['mode'].lower() == 'test', "change training mode to 'test' or 'eval'"
    assert params['beam_size'] == params['batch_size'], "Beam size must be same as batch_size"
    assert params['model'] == 'PGN', 'Please change the model to PGN'

    print('Building the model....')
    model = PGN(params)

    print('Creating vocab.....')
    print(params['vocab_size'])
    vocab = Vocab(params['vocab_path'], params['vocab_size'])
    print(vocab.count)

    print('Creating the batcher...')
    batch = batcher(vocab, params)

    print('Creating the checkpoint manager.......')
    checkpoint_dir = '{}/checkpoint'.format(params['pgn_model_dir'])
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print('Model restored')
    else:
        print('Initializing from scratch')

    result = []
    test_step = 0
    for b in batch:
        if test_step < 10:
            result.append(beam_decode(model, b, vocab, params))
            test_step += 1
        else:
            break
    return result


def test_and_save(params):
    assert params['test_save_dir'], "provide a dir where to save the results"

    result = test(params)
    # print("prediction result is: ", result)
    save_predict_result(result, params)


def save_predict_result(result, params):
    test_df = pd.read_csv(params['test_file_path'])
    test_df = test_df.loc[:9]
    test_df['Prediction'] = result
    test_df = test_df[['QID', 'Prediction']]
    print('Prediction result: ', test_df)
    test_df.to_csv(params['test_result_path'])
