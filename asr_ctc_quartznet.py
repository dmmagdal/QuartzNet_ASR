# asr_ctc_quartznet.py
# Replicate the original Quartznet paper by training a Quartznet 15x5
# model to perform automatic speech recognition (ASR) on the
# Librispeech dataset using CTC loss.
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


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
from quartznet import ASRCallbackEval, StringMap


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
	batch_size = 32
	train_dataset = tf.data.Dataset.from_tensor_slices(
		(list(df_train["file_name"]), 
		list(df_train["normalized_transcription"]))
	)
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
	)
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
	c_out = len(char_to_num.get_vocabulary())

	# Training variables.
	epochs = 1 # From Keras ASR_CTC example
	#epochs = 300 # From Quartznet paper 
	#epochs = 400 # From Quartznet paper (SOTA)
	learning_rate = 1e-4
	optimizer = tfa.optimizers.NovoGrad(learning_rate)
	val_callback = ASRCallbackEval(valid_dataset, num_to_char)

	# Initialize a Quartznet 15x5 model.
	cfg = Config(3)
	quartz_15x5 = QuartzNet(c_in, c_out, cfg, name="quartznet_15x5")
	quartz_15x5.build(input_shape=(None, None, c_in))
	quartz_15x5.summary()

	# Compile and train.
	quartz_15x5.compile(optimizer=optimizer, loss=CTCLoss)
	quartz_15x5.fit(
		train_dataset, epochs=epochs, callbacks=[val_callback]
	)

	# Inference.
	predictions = []
	targets = []
	for batch in valid_dataset:
		X, y = batch
		batch_predictions = quartz_15x5.predict(X)
		batch_predictions = decode_batch_predictions(batch_predictions)
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
	main()