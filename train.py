import os
import keras
import hashlib
import json
import numpy as np
from uuid import uuid4
from argparse import Namespace
from keras.layers import Dense, Dropout, Flatten, Input, Lambda, ELU, Permute
from keras.activations import selu
from keras.layers import Activation
from keras.layers.noise import AlphaDropout
from keras.layers.merge import Add, Multiply, Maximum
from keras.models import Model
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.optimizers import Nadam
from keras import backend as K

def mkdirs_exists_ok(path):
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise


class LossHistoryCallback(Callback):
  def __init__(self, history_path):
    self._history_path = history_path

  def on_epoch_end(self, epoch, logs={}):
    if not logs:
      return

    with open(self._history_path, "a") as f:
      line = "{}".format(logs["loss"])
      if "val_loss" in logs:
        line += ",{}".format(logs["val_loss"])
      line += "\n"
      f.write(line)

def generate_data(batch_size):
  hash_func = hashlib.md5
  input_bytes = 512 / 8
  batch_bytes = batch_size * input_bytes
  while True:
    s = os.urandom(batch_bytes)
    hashes = []
    for i in range(batch_size):
      hashes.append(hash_func(s[i * input_bytes:(i + 1) * input_bytes]).digest())

    x = 2 * np.unpackbits(
      np.frombuffer(s, dtype=np.uint8).reshape(batch_size, 64),
      axis=1).astype(np.float32) - 1.
    y = np.unpackbits(
      np.frombuffer(b"".join(hashes), dtype=np.uint8).reshape(batch_size, 16),
      axis=1)

    yield x, y


def binary_loss(y, x):
  return y * -K.minimum(1., x) + (1. - y) * K.maximum(-1., x) - K.minimum(1., K.square(x))

def binary_accuracy(y, x):
  sx = K.sign(x)
  return K.maximum(0., y * sx + (1. - y) * -sx)


def get_hash_model():
  relu = K.relu
  input_block = Input(shape=(512,), name="input")

  x = input_block
  x = Dense(1024, kernel_initializer="lecun_normal", activation="selu")(x)
  x = AlphaDropout(0.25)(x)

  for _ in range(50):
    before = x
    x = Dense(1024, kernel_initializer="lecun_normal", activation="selu")(x)
    x = AlphaDropout(0.25)(x)
    x = Add()([before, x])

  x = Dense(128)(x)
  outputs = [x]

  model = Model(inputs=[input_block], outputs=outputs)
  model.compile(optimizer=Nadam(lr=3e-3, clipnorm=1e-3), loss=binary_loss, metrics=[binary_accuracy])
  return model


def train():
  args = Namespace()
  args.validation_steps = 1e3
  args.samples_per_epoch = int(1e6)
  args.total_samples  = int(1e8)
  args.batch_size = 256
  args.save_weights = False
  args.train_dir = None

  do_validation = args.validation_steps > 0
  initial_epoch = 1

  batches_per_epoch = int(round(args.samples_per_epoch / args.batch_size))
  epochs = int(np.ceil(1. * args.total_samples / args.samples_per_epoch))

  training_id = str(uuid4())
  train_dir = args.train_dir or "./trained/{}".format(training_id)
  checkpoint_path_fmt = os.path.join(train_dir, "weights.{epoch:03d}.keras")
  mkdirs_exists_ok(train_dir)
  history_path = os.path.join(train_dir, "history.csv")
  print("Writing train results to {}".format(train_dir))
  with open(os.path.join(train_dir, "args.json"), "wb") as f:
    json.dump(vars(args), f)

  model = get_hash_model()
  model.summary()

  model.fit_generator(
    generate_data(args.batch_size),
    batches_per_epoch,
    epochs=epochs,
    validation_data=(generate_data(args.batch_size) if args.validation_steps else None),
    validation_steps=args.validation_steps,
    max_queue_size=30,
    verbose=1,
    initial_epoch=initial_epoch,
    callbacks=[
      TerminateOnNaN(),
      LossHistoryCallback(history_path),
    ] + (  #
      [ModelCheckpoint("{}/{}".format(train_dir, checkpoint_path_fmt))]
      if args.save_weights else []  #
    ) + ([
      EarlyStopping(patience=120),
      ReduceLROnPlateau(factor=0.6, patience=20, verbose=1),
    ] if do_validation else []))

def main():
  train()

if __name__ == "__main__":
  main()
