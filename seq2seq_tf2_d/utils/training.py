import tensorflow as tf
from seq2seq_tf2_d.models.seq2seq import SequenceToSequence
from seq2seq_tf2_d.utils.batcher_utils import  batcher, Vocab
from seq2seq_tf2_d.utils.train_helper import train_model


def train(params):
    global checkpoint_dir, ckpt, model
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    print('true vocab is ', vocab)

    print("Creating the batcher ...")
    dataset = batcher(vocab, params) #下次PGN详细讲，仅仅是将数据封装成tf特定的格式
    print("Building the model ...")
    if params["model"] == "SequenceToSequence":
        model = SequenceToSequence(params)
    # elif params["model"] == "PGN":
    #     model = PGN(params)

    print("Creating the checkpoint manager")
    #基本为固定写法
    if params["model"] == "SequenceToSequence":
        #每训练一定轮次，就保存一次
        checkpoint_dir = "{}/checkpoint".format(params["seq2seq_model_dir"])
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), SequenceToSequence=model)
    # elif params["model"] == "PGN":
    #     checkpoint_dir = "{}/checkpoint".format(params["pgn_model_dir"])
    #     ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    print("Starting the training ...")
    train_model(model, dataset, params, ckpt_manager,vocab)


if __name__ == '__main__':
    pass