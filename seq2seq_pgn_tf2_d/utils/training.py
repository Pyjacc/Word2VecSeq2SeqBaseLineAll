import tensorflow as tf
from seq2seq_pgn_tf2_d.utils.batcher_utils import batcher
from seq2seq_pgn_tf2_d.utils.train_helper import train_model
from seq2seq_pgn_tf2_d.models.pgn import PGN
from seq2seq_pgn_tf2_d.utils.vocab_load import Vocab


def train(params):
    global checkpoint_dir, ckpt, model
    assert params["mode"].lower() == "train", "change training mode to 'train'"
    assert params['model'] == "PGN", "change model to PGN to train"

    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    print('true vocab is ', vocab)

    print("Creating the batcher ...")
    dataset = batcher(vocab, params)
    print("Building the model ...")
    if params["model"] == "PGN":
        model = PGN(params)
    # elif params["model"] == "SequenceToSequence":
    #     model = SequenceToSequence(params)

    print("Creating the checkpoint manager")
    #基本为固定写法
    if params["model"] == "PGN":
        checkpoint_dir = "{}/checkpoint".format(params["pgn_model_dir"])
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
    # elif params["model"] == "SequenceToSequence":
    #     checkpoint_dir = "{}/checkpoint".format(params["seq2seq_model_dir"])
    #     ckpt = tf.train.Checkpoint(step=tf.Variable(0), SequenceToSequence=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    print("Starting the training ...")
    train_model(model, dataset, params, ckpt_manager, vocab)


if __name__ == '__main__':
    pass