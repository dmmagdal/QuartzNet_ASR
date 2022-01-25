# asr_ctc_quartznet.py
# Replicate the original Quartznet paper by training a Quartznet 15x5
# model to perform automatic speech recognition (ASR) on the
# Librispeech dataset using CTC loss.
# Tensorflow 2.7
# Python 3.7
# Windows/MacOS/Linux


import os
import string
import numpy as np
import pandas as pd
import tensorflow_addons as tfa
#import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from IPython import display
from jiwer import wer
from config import Config
from quartznet import QuartzNet, CTCLoss, decode_batch_predictions
from quartznet import ASRCallbackEval, StringMap, CTCNNLoss, EpochSave


def main():
	# Download librispeech dataset. For manual downloads, the following
	# URLs are for the respective US mirrors:
	# dev-clean.tar.gz [337M] (development set, "clean" speech)
	# URL: https://us.openslr.org/resources/12/dev-clean.tar.gz
	# dev-other.tar.gz [314M] (development set, "other", more
	# challenging, speech)
	# URL: https://us.openslr.org/resources/12/dev-other.tar.gz
	# test-clean.tar.gz [346M] (test set, "clean" speech )
	# URL: https://us.openslr.org/resources/12/test-clean.tar.gz
	# test-other.tar.gz [328M] (test set, "other" speech )
	# URL: https://us.openslr.org/resources/12/test-other.tar.gz
	# train-clean-100.tar.gz [6.3G] (training set of 100 hours "clean"
	# speech )
	# URL: https://us.openslr.org/resources/12/train-clean-100.tar.gz
	# train-clean-360.tar.gz [23G] (training set of 360 hours "clean"
	# speech )
	# URL: https://us.openslr.org/resources/12/train-clean-360.tar.gz
	# train-other-500.tar.gz [30G] (training set of 500 hours "other"
	# speech )
	# URL: https://us.openslr.org/resources/12/train-other-500.tar.gz
	#
	# Using tensorflow dataset module, you can use the following code
	# to download all of the above data (check for the local cache
	# where tensorflow datasets stores this data to) which can take up
	# at least the sum of the data listed for each (6.3 + 23 + 30 =
	# 59GB total).
	#librispeech_ds = tfds.load(
	#	"librispeech", split="dev-clean", try_gcs=True
	#)
	# Important reference for using tensorflow datasets to download
	# data: https://www.projectpro.io/recipes/import-inbuilt-dataset-
	# tensorflow

	# QuartzNet trained for 300 epochs. Here are the respective greedy
	# WER (%) on the following librispeech datasets:
	# Model | dev-clean | dev-other
	#  5x5      5.39        15.69
	#  10x5     4.14        12.33
	#  15x5     3.98        11.58

	# Download LJSpeech dataset.
	data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
	data_path = keras.utils.get_file("LJSpeech-1.1", data_url, untar=True)
	#data_path = "./LJSpeech-1.1"
	wavs_path = data_path + "/wavs/"
	metadata_path = data_path + "/metadata.csv"

	metadata_df = pd.read_csv(
		metadata_path, sep="|", header=None, quoting=3
	)
	metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
	metadata_df = metadata_df[["file_name", "normalized_transcription"]]
	metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
	print(metadata_df.head(3))

	# Split the dataset into training and validation set.
	split = int(len(metadata_df) * 0.90)
	df_train = metadata_df[:split]
	df_valid = metadata_df[split:]
	print(f"Size of training set: {len(df_train)}")
	print(f"Size of vaidating set: {len(df_valid)}")

	# Initialize the character mappings to their values and visa versa. 
	chars = list(string.ascii_lowercase + "?!' ")
	char_to_num = StringMap(vocabulary=chars, oov_token="")
	num_to_char = StringMap(
		vocabulary=char_to_num.get_vocabulary(), oov_token="", 
		invert=True
	)
	char_to_num = tf.keras.layers.StringLookup(
		vocabulary=chars, oov_token=""
	)
	num_to_char = tf.keras.layers.StringLookup(
		vocabulary=char_to_num.get_vocabulary(), oov_token="", 
		invert=True
	)
	'''
	char_to_num = tf.lookup.StaticHashTable(
		initializer=tf.lookup.KeyValueTensorInitializer(
			keys=tf.constant([u for i, u in enumerate(chars)]),
			values=tf.constant([i for i, u in enumerate(chars)]),
		),
		default_value=tf.constant(" "),
	)
	num_to_char = tf.lookup.StaticHashTable(
		initializer=tf.lookup.KeyValueTensorInitializer(
			keys=tf.constant([i for i, u in enumerate(chars)]),
			values=tf.constant([u for i, u in enumerate(chars)]),
		),
		default_value=tf.constant(" "),
	)
	'''
	print(
		f"The vocabulary is: {char_to_num.get_vocabulary()}"
		f" (size = {char_to_num.vocabulary_size()})"
	)

	# Variables regarding converting the audio to mel spectrograms.
	frame_length = 256
	frame_step = 160
	n_fft = 384 # From Keras ASR_CTC example
	#n_fft = 1024

	def encode_single_sample(wav_file, label):
		# Process the audio.
		file = tf.io.read_file(wavs_path + wav_file + ".wav")
		audio, _ = tf.audio.decode_wav(file)
		audio = tf.squeeze(audio, axis=-1)
		audio = tf.cast(audio, tf.float32)

		spectrogram = tf.signal.stft(
			audio, frame_length=frame_length, frame_step=frame_step,
			fft_length=n_fft
		)
		spectrogram = tf.abs(spectrogram)
		spectrogram = tf.math.pow(spectrogram, 0.5)

		means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
		stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
		spectrogram = (spectrogram - means) / (stddevs + 1e-10)

		# Process the label.
		label = tf.strings.lower(label)
		label = tf.strings.unicode_split(label, input_encoding="UTF-8")
		label = char_to_num(label)
		return spectrogram, label

	# Create dataset objects.
	batch_size = 8#32
	train_dataset = tf.data.Dataset.from_tensor_slices(
		(list(df_train["file_name"]), 
		list(df_train["normalized_transcription"]))
	)#.take(batch_size * 10)
	train_dataset = (
		train_dataset.map(
			encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
		)
		.padded_batch(batch_size)
		.prefetch(buffer_size=tf.data.AUTOTUNE)
	)
	valid_dataset = tf.data.Dataset.from_tensor_slices(
		(list(df_valid["file_name"]), 
		list(df_valid["normalized_transcription"]))
	)#.take(batch_size * 10)
	valid_dataset = (
		valid_dataset.map(
			encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
		)
		.padded_batch(batch_size)
		.prefetch(buffer_size=tf.data.AUTOTUNE)
	)

	# Model input and output parameters. Input is mel-spectrograms and
	# output is a character from the vocabulary.
	c_in = n_fft // 2 + 1
	#c_out = len(char_to_num.get_vocabulary())
	c_out = char_to_num.vocabulary_size()

	# Training variables.
	#epochs = 1 # From Keras ASR_CTC example
	epochs = 50 # From Keras ASR_CTC example (recommended min epochs)
	#epochs = 300 # From Quartznet paper 
	#epochs = 400 # From Quartznet paper (SOTA)
	learning_rate = 1e-6
	optimizer = tfa.optimizers.NovoGrad(learning_rate)
	val_callback = ASRCallbackEval(valid_dataset, num_to_char)
	model_name = "quartznet_15x5"

	# Test custom training pause/resuming.
	#epochs = 10

	# Set checkpoint path and initialize ModelCheckpoint callback.
	checkpoint_dir = model_name + "_checkpoints_weights_only/"
	checkpoint = keras.callbacks.ModelCheckpoint(
		checkpoint_dir + "cp-{epoch:04d}.ckpt",
		verbose=1,
		save_weights_only=True,
	)

	if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
		# Load model weights with tf.train.latest_checkpoint.
		latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
		initial_epoch = int(
			latest_checkpoint.lstrip(checkpoint_dir + "cp-").rstrip(".ckpt")
		)
	else:
		# No previous checkpoint exits. Create checkpoint directory and
		# start at the first epoch.
		os.makedirs(checkpoint_dir, exist_ok=True)
		latest_checkpoint = None
		initial_epoch = 0

	# Initialize a Quartznet 15x5 model.
	cfg = Config(3) # Config for a Quartznet 15x5 model.
	quartz_15x5 = QuartzNet(c_in, c_out + 1, cfg, name=model_name)
	quartz_15x5.build(input_shape=(None, None, c_in))
	quartz_15x5.summary()

	# Compile model.
	#quartz_15x5.compile(optimizer=optimizer, loss=CTCLoss)
	quartz_15x5.compile(optimizer=optimizer, loss=CTCNNLoss)

	if latest_checkpoint:
		# Re-initialize a Quartznet 15x5 model (and load the previous
		# checkpoint weights). Be sure to recompile the model
		# (optimizer is re-initialized and is therefore different from
		# optimizer state in previous run).
		print("Loading checkpoint: {}".format(latest_checkpoint))
		quartz_15x5 = QuartzNet(c_in, c_out + 1, cfg, name=model_name)
		quartz_15x5.load_weights(latest_checkpoint).expect_partial()
		quartz_15x5.build(input_shape=(None, None, c_in))
		quartz_15x5.summary()
		quartz_15x5.compile(optimizer=optimizer, loss=CTCNNLoss)
	else:
		# Initialize a Quartznet 15x5 model and compile it.
		print("Starting training from beginning...")
		quartz_15x5 = QuartzNet(c_in, c_out + 1, cfg, name=model_name)
		quartz_15x5.build(input_shape=(None, None, c_in))
		quartz_15x5.summary()
		quartz_15x5.compile(optimizer=optimizer, loss=CTCNNLoss)

	# Train model (with initial epoch value set) and save (its weights
	# once finished training)
	quartz_15x5.fit(
		train_dataset, 
		validation_data=valid_dataset, 
		callbacks=[val_callback, checkpoint],
		epochs=epochs,
		initial_epoch=initial_epoch,
	)
	save_path = checkpoint_dir + "final_weights_epoch_{epochs:04d}/"
	os.makedirs(save_path, exist_ok=True)
	quartz_15x5.save_weights(save_path)
	
	# Load saved model weights (must be same 15x5 configuration). This
	# model can be re built (call build() with correct input_shape) and
	# compiled (call compile with same optimizer and loss function as
	# before) to be further trained.
	cfg = Config(3)
	loaded_model = QuartzNet(c_in, c_out + 1, cfg, name=model_name)
	loaded_model.load_weights(save_path).expect_partial()
	loaded_model.build(input_shape=(None, None, c_in))
	loaded_model.summary()
	loaded_model.compile(optimizer=optimizer, loss=CTCNNLoss)

	'''
	data = list(train_dataset.as_numpy_iterator())[0]
	input_sample = data[0]
	label_output = data[1]
	print("data")
	print(data) # Combined data (Input, Output). Only 1 batch long.
	print(len(data))
	print("Input Mel-Spec")
	print(input_sample) # Input to the model (Mel-Spec). Shape (batch_size, time, n_fft // 2 + 1)
	print(type(input_sample))
	print(input_sample.shape) # (32, 1358, 193)
	print("Ground truth labels")
	print(label_output) # Output from the model (char). Shape (batch_size, text_length)
	print(type(label_output))
	print(label_output.shape) # (32, 163)
	pred = quartz_15x5.predict(train_dataset)
	print("Model output predictions")
	print(pred) # Output prediction from the model (char). Shape (batch_size, , characters)
	print(type(pred)) 
	print(tf.shape(pred)) # (32, 679, 31)
	print()
	print(pred[0])

	print("calculate ctc loss")
	print("tf.nn.ctc_loss arguments")
	label_lengths = tf.cast(tf.shape(label_output)[1], dtype="int64")
	label_lengths = label_lengths * tf.ones(shape=(batch_size), dtype="int64")
	logit_lengths = tf.cast(
		tf.math.ceil(tf.shape(pred)[1] / quartz_15x5.feature_time_reduction_factor), 
		dtype="int64"
	)
	logit_lengths = logit_lengths * tf.ones(shape=(batch_size), dtype="int64")
	print(label_lengths)
	print(logit_lengths)
	loss = tf.nn.ctc_loss(
		label_output, pred, label_length=label_lengths, logit_length=logit_lengths,
		blank_index=-1, 
		logits_time_major=False
	)
	print(loss)
	exit()
	'''

	# Inference.
	predictions = []
	targets = []
	for batch in valid_dataset:
		X, y = batch
		batch_predictions = quartz_15x5.predict(X)
		batch_predictions = decode_batch_predictions(batch_predictions, num_to_char)
		predictions.extend(batch_predictions)
		for label in y:
			label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("UTF-8")
			targets.append(label)
	wer_score = wer(targets, predictions)
	print("-" * 100)
	print(f"Word Error Rate: {wer_score:.4f}")
	print("-" * 100)
	for i in np.random.randint(0, len(predictions), 5):
		print(f"Target    : {targets[i]}")
		print(f"Prediction: {predictions[i]}")
		print("-" * 100)

	# Exit the program.
	exit(0)


if __name__ == "__main__":
	with tf.device("/cpu:0"):
		main()
	#main()