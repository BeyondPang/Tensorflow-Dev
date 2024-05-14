#import necessary libs
import io
import re
import string
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 1024
BUFFER_SIZE = 10000

#Prepare data
path_to_zip = tf.keras.utils.get_file('shakespears.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
with open(path_to_zip) as f:
    lines = f.read().splitlines()
text_ds = tf.data.Dataset.from_tensor_slices(lines).filter(lambda x: tf.cast(tf.strings.length(x), bool))
for line in text_ds.take(20):
    print(line.numpy().decode('utf-8'))

def custom_standardization(input_string):
  lowercase = tf.strings.lower(input_string)
  return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')

vocab_size = 4096
sequence_length = 10
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)
vectorize_layer.adapt(text_ds.batch(1024))
inverse_vocab = vectorize_layer.get_vocabulary()
print(inverse_vocab[:10])

text_vector_ds = text_ds.batch(1024).map(vectorize_layer).unbatch()
sequences = list(text_vector_ds.as_numpy_iterator())
print(len(sequences))
for sequence in sequences[:3]:
    print(f'{sequence} --> {[inverse_vocab[i] for i in sequence]}')

num_ns = 4

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  targets, contexts, labels = [], [], []
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
  for sequence in tqdm.tqdm(sequences):
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
        sequence,
        vocabulary_size=vocab_size,
        sampling_table=sampling_table,
        window_size=window_size,
        negative_samples=0)
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=SEED,
          name="negative_sampling")
      context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0] * num_ns, dtype="int64")

      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

targets, contexts, labels = generate_training_data(sequences, window_size=2, num_ns=4, vocab_size=vocab_size, seed=SEED)

targets = np.array(targets)
contexts = np.array(contexts)
labels = np.array(labels)
print(targets.shape, contexts.shape, labels.shape)
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)  

class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(vocab_size, embedding_dim, name='w2v_embedding')
    self.context_embedding = layers.Embedding(vocab_size, embedding_dim)

  def call(self, pair):
    target, context = pair
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    word_emb = self.target_embedding(target)
    context_emb = self.context_embedding(context)
    dot_product = tf.einsum('be,bce->bc', word_emb, context_emb)
    return dot_product

embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])