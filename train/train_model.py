import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#from keras.layers import LSTM
#from keras.callbacks import ModelCheckpoint
#from keras.utils import np_utils
import pickle
import tflearn
from tflearn.data_utils import *
from random import randint

import re
import sys
import numpy as np
import tensorflow as tf

from tflearn.helpers.trainer import Trainer, evaluate as eval
from tflearn.helpers.evaluator import Evaluator
from tflearn.utils import feed_dict_builder, is_none


class SGenerator(object):
    def __init__(self, network, dictionary=None, seq_maxlen=25,
                 clip_gradients=0.0, tensorboard_verbose=0,
                 tensorboard_dir="/tmp/tflearn_logs/",
                 checkpoint_path=None, max_checkpoints=None,
                 session=None):
        assert isinstance(network, tf.Tensor), "'network' arg is not a Tensor!"
        self.net = network
        self.train_ops = tf.get_collection(tf.GraphKeys.TRAIN_OPS)
        self.trainer = Trainer(self.train_ops,
                               clip_gradients=clip_gradients,
                               tensorboard_dir=tensorboard_dir,
                               tensorboard_verbose=tensorboard_verbose,
                               checkpoint_path=checkpoint_path,
                               max_checkpoints=max_checkpoints,
                               session=session)
        self.session = self.trainer.session
        self.inputs = tf.get_collection(tf.GraphKeys.INPUTS)
        self.targets = tf.get_collection(tf.GraphKeys.TARGETS)
        self.predictor = Evaluator([self.net],
                                   session=self.session)
        self.dic = dictionary
        self.rev_dic = reverse_dictionary(dictionary)
        self.seq_maxlen = seq_maxlen

    def fit(self, X_inputs, Y_targets, n_epoch=10, validation_set=None,
            show_metric=False, batch_size=None, shuffle=None,
            snapshot_epoch=True, snapshot_step=None, excl_trainops=None,
            run_id=None):
        if batch_size:
            for train_op in self.train_ops:
                train_op.batch_size = batch_size

        valX, valY = None, None
        if validation_set:
            if isinstance(validation_set, float):
                valX = validation_set
                valY = validation_set
            else:
                valX = validation_set[0]
                valY = validation_set[1]

        # For simplicity we build sync dict synchronously but
        # Trainer support asynchronous feed dict allocation
        feed_dict = feed_dict_builder(X_inputs, Y_targets, self.inputs,
                                      self.targets)
        feed_dicts = [feed_dict for i in self.train_ops]

        val_feed_dicts = None
        if not (is_none(valX) or is_none(valY)):
            if isinstance(valX, float):
                val_feed_dicts = valX
            else:
                val_feed_dict = feed_dict_builder(valX, valY, self.inputs,
                                                  self.targets)
                val_feed_dicts = [val_feed_dict for i in self.train_ops]

        # Retrieve data preprocesing and augmentation
        dprep_dict, daug_dict = {}, {}
        dprep_collection = tf.get_collection(tf.GraphKeys.DATA_PREP)
        daug_collection = tf.get_collection(tf.GraphKeys.DATA_AUG)
        for i in range(len(self.inputs)):
            if dprep_collection[i] is not None:
                dprep_dict[self.inputs[i]] = dprep_collection[i]
            if daug_collection[i] is not None:
                daug_dict[self.inputs[i]] = daug_collection[i]

        self.trainer.fit(feed_dicts, val_feed_dicts=val_feed_dicts,
                         n_epoch=n_epoch,
                         show_metric=show_metric,
                         snapshot_step=snapshot_step,
                         snapshot_epoch=snapshot_epoch,
                         shuffle_all=shuffle,
                         dprep_dict=dprep_dict,
                         daug_dict=daug_dict,
                         excl_trainops=excl_trainops,
                         run_id=run_id)
        self.predictor = Evaluator([self.net],
                                   session=self.trainer.session)

    def _predict(self, X):
        feed_dict = feed_dict_builder(X, None, self.inputs, None)
        return self.predictor.predict(feed_dict)

    def generate(self, seq_length, temperature=0.5, seq_seed=None,
                 display=False):
        generated = seq_seed[:]
        sequence = seq_seed[:]
        whole_sequence = seq_seed[:]

        if display: sys.stdout.write(str(generated))

        for i in range(seq_length):
            x = np.zeros((1, self.seq_maxlen, len(self.dic)))
            for t, char in enumerate(sequence):
                x[0, t, self.dic[char]] = 1.

            preds = self._predict(x)[0]
            next_index = _sample(preds, temperature)
            next_char = self.rev_dic[next_index]

            if type(sequence) == str:
                generated += next_char
                sequence = sequence[1:] + next_char
                whole_sequence += next_char
            else:
                generated.append(next_char)
                sequence = sequence[1:]
                sequence.append(next_char)
                whole_sequence.append(next_char)

            if display:
                sys.stdout.write(str(next_char))
                sys.stdout.flush()

        if display: print()

        return whole_sequence

    def save(self, model_file):
        self.trainer.save(model_file)

    def load(self, model_file, **optargs):
        self.trainer.restore(model_file, **optargs)
        self.session = self.trainer.session
        self.predictor = Evaluator([self.net],
                                   session=self.session,
                                   model=None)
        for d in tf.get_collection(tf.GraphKeys.DATA_PREP):
            if d: d.restore_params(self.session)

    def get_weights(self, weight_tensor):
        return weight_tensor.eval(self.trainer.session)

    def set_weights(self, tensor, weights):
        op = tf.assign(tensor, weights)
        self.trainer.session.run(op)

    def evaluate(self, X, Y, batch_size=128):
        feed_dict = feed_dict_builder(X, Y, self.inputs, self.targets)
        return eval(self.trainer.session, self.net, feed_dict, batch_size)


def reverse_dictionary(dic):
    # Build reverse dict
    rev_dic = {}
    for key in dic:
        rev_dic[dic[key]] = key
    return rev_dic


def _sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.log(preds) / temperature
    preds = np.exp(preds) / np.sum(np.exp(preds))
    return np.argmax(np.random.multinomial(1, preds, 1))

def generate_encoding_representations(encoding):
    chars = set([z for y in encoding.values() for z in y.split(" ")])
    chars.add("<w>")
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return char_indices, indices_char

def generate_numpy_representation(file_num, encoding,char_indices, maxlen=30, step=5):
    print("Opening ../data/small_files/data%s" % file_num )
    text = open("../data/small_files/data%s" % file_num,"r")
    words = [w for s in text for w in s.strip().split() if w]
    numbers = []
    for word in words:
        # FIX THIS!!! WHITESPACES CAUSES INCONSISTENCE IN ENCODING
        if word in encoding:
            wordpieces = encoding[word]
            indices = [char_indices[a] for a in wordpieces.split(" ")]
            indices.append(char_indices["<w>"])
            numbers.extend(indices)

        else:
            print("Missing encoding: %s" % word)

    sentences = []
    next_chars = []
    for i in range(0, len(numbers) - maxlen, step):
        sentences.append(numbers[i: i + maxlen])
        next_chars.append(numbers[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(char_indices.keys())), dtype=np.bool)
    y = np.zeros((len(sentences), len(char_indices.keys())), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char] = 1
        y[i, next_chars[i]] = 1
    return X,y

def generate_sample_seed(X,maxlen,indices_char):
    seed = []
    idx = randint(0,len(X))
    for i in range(maxlen):
        char_i = np.argmax(X[idx][i])
        seed.append(indices_char[char_i])
    return seed

if __name__ == "__main__":
    encoding = pickle.load(open("../preprocessing/word_pieces.dict_new_8000", "rb"))
    #encoding = pickle.load(open("../preprocessing/data5.dict", "rb"))

    char_indices, indices_char = generate_encoding_representations(encoding)

    char_idx = len(char_indices)
    maxlen = 20

    g = tflearn.input_data([None, maxlen, len(char_indices)])
    g = tflearn.lstm(g, 1024, return_seq=True)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.lstm(g, 1024, return_seq=True)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.lstm(g, 1024)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.fully_connected(g, char_idx, activation='softmax')
    g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                           learning_rate=0.001)

    m = SGenerator(g, dictionary=char_indices, seq_maxlen=maxlen, clip_gradients=5.0)
    m.load("posledni_model_1024")

    i=582

    while True:
        print("Epocha: %s" % i)
        i+=1
        dataset_idx = (i % 135)
     #   X, Y = generate_numpy_representation(5, encoding, char_indices, maxlen)
        X, Y = generate_numpy_representation(dataset_idx, encoding, char_indices, maxlen)
        seed = generate_sample_seed(X,maxlen,indices_char)
        m.fit(X, Y, validation_set=0.1, batch_size=128,
              n_epoch=1, run_id='receptron')
        m.save("posledni_model_1024")
        print("-- TESTING...")
        print("\n-- Test with temperature of 1.0 --")
      #  import pdb;pdb.set_trace()
        ddd = m.generate(400, temperature=1.0, seq_seed=seed)
        print("Seed: %s" % re.sub("</w>", " ", "".join(seed)))
        print(re.sub("<w>", " ", "".join(ddd)))
        print("\n-- Test with temperature of 0.75 --")
        ddd = m.generate(400, temperature=0.75, seq_seed=seed)
        print(re.sub("<w>", " ", "".join(ddd)))
        print("\n-- Test with temperature of 0.5 --")
        ddd = m.generate(400, temperature=0.5, seq_seed=seed)
        print(re.sub("<w>", " ", "".join(ddd)))
        print("\n-- Test with temperature of 0.25 --")
        ddd = m.generate(400, temperature=0.25, seq_seed=seed)
        print(re.sub("<w>", " ", "".join(ddd)))
    import pdb; pdb.set_trace()
