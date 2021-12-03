# quartznet.py
# Tensorflow implementation of the QuartzNet model for Automatic Speech
# Recognition (ASR). Potential use in neural TTS within TalkNet 1/2.
# Source: https://gitlab.com/Jaco-Assistant/Scribosermo/
# Source: https://github.com/stefanpantic/asr
# Source: https://arxiv.org/pdf/1910.10261.pdf
# Tensorflow 2.4.0
# Python 3.7
# Windows/MacOS/Linux


import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from jiwer import wer
from typing import List


class Conv1DBlock(layers.Layer):
	def __init__(self, filters, kernel_size, use_bias=False, **kwargs):
		super().__init__(**kwargs)

		self.conv = layers.Conv1D(
			filters=filters, kernel_size=kernel_size, 
			data_format="channels_last", kernel_regularizer=None,
			use_bias=use_bias
		)
		self.batch_norm = layers.BatchNorm(momentum=0.9)
		self.relu = layers.ReLU()


	def call(self, inputs):
		x = self.conv(inputs)
		x = self.batch_norm(x)
		output = self.relu(x)
		return output


class SepConv1DBlock(layers.Layer):
	def __init__(self, filters, kernel_size, use_bias=False, **kwargs):
		super().__init__(**kwargs)

		self.conv = layers.SeparableConv1D(
			filters=filters, kernel_size=kernel_size, 
			data_format="channels_last", kernel_regularizer=None,
			use_bias=use_bias
		)
		self.batch_norm = layers.BatchNorm(momentum=0.9)
		self.relu = layers.ReLU()


	def call(self, inputs):
		x = self.conv(inputs)
		x = self.batch_norm(x)
		output = self.relu(x)
		return output


class Module(layers.Layer):
	def __init__(self, filters, kernel_size, has_relu=True, **kwargs):
		super().__init__(**kwargs)

		pad = int(math.floor(kernel_size / 2))
		self.pad1d = layers.ZeroPadding1D(padding=(pad, pad))
		self.sep_conv1d = layers.SeparableConv1D(
			filters=filters, kernel_size=kernel_size, padding="valid",
			data_format="channels_last", depthwise_regularizer=None,
			pointwise_regularizer=None, use_bias=False,
		)
		self.batch_norm = layers.BatchNormalization(momentum=0.5)

		self.has_relu = has_relu
		if self.has_relu:
			self.relu = layers.ReLU()


	def call(self, inputs):
		x = self.pad1d(inputs)
		x = self.sep_conv1d(x)
		x = self.batch_norm(x)

		if self.has_relu:
			x = self.relu(x)
		return x


class BaseBlock(layers.Layer):
	def __init__(self, filters, kernel_size, repeat, **kwargs):
		super().__init__(**kwargs)

		self.model = keras.Sequential()
		for _ in range(repeat - 1):
			layer = Module(filters=filters, kernel_size=kernel_size)
			self.model.add(layer)
		last_layer = Module(
			filters=filters, kernel_size=kernel_size, has_relu=False
		)
		self.model.add(last_layer)

		self.res_conv = layers.Conv1D(
			filters=filters, kernel_size=1, padding="valid",
			kernel_regularizer=None, use_bias=False,
		)
		self.res_batch_norm = layers.BatchNormalization(momentum=0.5)

		self.relu = layers.ReLU()


	def call(self, inputs):
		block_output = self.model(inputs)

		# Residual passed through pointwise conv1D and batch norm.
		res_output = self.res_conv(inputs)
		res_output = self.res_batch_norm(res_output)

		# Add residual to the block output.
		sum_output = layers.Add()([block_output, res_output])

		# Pass the sum to the final relu.
		output = self.relu(sum_output)
		return output


class QuartzNet(keras.Model):
	def __init__(self, c_input, c_output, config, **kwargs):
		super().__init__(**kwargs)

		block_params = [
			[256, 33],
			[256, 39],
			[512, 51],
			[512, 63],
			[512, 75],
		]
		assert config.module_repeat > 0

		self.feature_time_reduction_factor = 2

		# Layers.
		# Padding layer.
		self.pad_1 = layers.ZeroPadding1D(padding=(16, 16))

		# First Conv1D-BatchNorm-Relu block.
		self.sep_conv1 = layers.SeparableConv1D(
			filters=256, kernel_size=33, strides=2, padding="valid",
			data_format="channels_last", depthwise_regularizer=None,
			pointwise_regularizer=None,
		)
		self.batch_norm_1 = layers.BatchNormalization(momentum=0.5)
		self.relu_1 = layers.ReLU()
		self.dropout_1 = layers.Dropout(0.1)

		# Next are the sequence of main blocks.
		blocks = []
		for filters, kernel_size in block_params:
			for _ in range(config.block_repeat):
				blocks.append(
					BaseBlock(filters, kernel_size, config.module_repeat)
				)
		self.blocks_model = keras.Sequential(blocks)

		# Followed by the remaining convolutional layers.
		self.pad_2 = layers.ZeroPadding1D(padding=(86, 86))
		self.sep_conv2 = layers.SeparableConv1D(
			filters=512, kernel_size=87, dilation_rate=2,
			padding="valid", data_format="channels_last",
			depthwise_regularizer=None, pointwise_regularizer=None,
			use_bias=False,
		)
		self.batch_norm_2 = layers.BatchNormalization(momentum=0.9)
		self.relu_2 = layers.ReLU()

		self.conv3 = layers.Conv1D(
			filters=1024, kernel_size=1, padding="valid",
			data_format="channels_last", kernel_regularizer=None,
			use_bias=False,
		)
		self.batch_norm_3 = layers.BatchNormalization(momentum=0.9)
		self.relu_3 = layers.ReLU()

		self.conv4 = layers.Conv1D(
			filters=c_output, kernel_size=1, padding="valid",
			data_format="channels_last", kernel_regularizer=None,
			use_bias=True,
		)
		#self.softmax = layers.Softmax()


	@tf.function(experimental_relax_shapes=True)
	def call(self, inputs, training=False):
		x = tf.identity(inputs)
		x = self.pad_1(x)
		x = self.sep_conv1(x)
		x = self.batch_norm_1(x)
		x = self.relu_1(x)
		x = self.dropout_1(x)
		x = self.blocks_model(x)
		x = self.pad_2(x)
		x = self.sep_conv2(x)
		x = self.batch_norm_2(x)
		x = self.relu_2(x)
		x = self.conv3(x)
		x = self.batch_norm_3(x)
		x = self.relu_3(x)
		x = self.conv4(x)
		x = tf.cast(x, dtype="float32")
		#x = self.softmax(x)
		x = tf.nn.log_softmax(x)
		output = tf.identity(x, name="output")
		return output


	def get_time_reduction_factor(self):
		return self.feature_time_reduction_factor


def CTCLoss(y_true, y_pred):
	# Compute the training-time loss value.
	batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
	input_length = tf.cast(tf.shape(y_pred)[0], dtype="int64")
	label_length = tf.cast(tf.shape(y_true)[0], dtype="int64")

	input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
	label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

	loss = keras.backend.ctc_batch_cost(
		y_true, y_pred, input_length, label_length
	)
	return loss


def decode_batch_predictions(pred, num_to_char):
	input_len = np.ones(pred.shape[0]) * pred.shape[1]

	# Use greedy search. For complex tasks, use beam search.
	results = keras.backend.ctc_decode(
		pred, input_length=input_length, greedy=True
	)[0][0]

	# Iterate over the results and get back the text.
	output_text = []
	for result in results:
		result = tf.strings.reduce_join(
			num_to_char(result),
		).numpy().decode('utf-8')
		output_text.append(result)
	return output_text


class ASRCallbackEval(keras.callbacks.Callback):
	# Display a batch of outputs after every epoch.
	def __init__(self, dataset, num_to_char):
		super().__init__()
		self.dataset = dataset
		self.num_to_char = num_to_char


	def on_epoch_end(self, epoch, logs=None):
		predictions = []
		targets = []
		for batch in self.dataset:
			X, y = batch
			batch_predictions = model.predict(X)
			batch_predictions = decode_batch_predictions(
				batch_predictions, self.num_to_char
			)
			predictions.extend(batch_predictions)
			for label in y:
				label = (
					tf.strings.reduce_join(self.num_to_char(label)).numpy().decode("utf-8")
				)
				targets.append(label)
			wer_score = wer(targets, predictions)
			print("-"*100)
			print(f"Word Error Rate: {wer_score:.4f}")
			print("-"*100)
			for i in np.random.randint(0, len(predictions), 2):
				print(f"Target     : {targets[i]}")
				print(f"Prediction : {predictions[i]}")
				print("-"*100)


class StringMap:
	# Simple replacement for tf.keras.layers.StringLookup for those not
	# running the latest tensorflow 2 version.
	def __init__(self, vocabulary:List[str], oov_token="[UNK]", invert=False):
		# Check that all elements of the vocabulary are unique.
		if len(vocabulary) != len(list(set(vocabulary))):
			raise ValueError("vocabulary argument must be a list of unique values")

		# Check that the vocabulary list doesn't already have the oov
		# token.
		if oov_token in vocabulary:
			vocabulary.remove(oov_token)
			#raise ValueError("oov_token argument value cannot be a part of the vocabulary")

		self.oov_token = oov_token

		# Combine the vocabulary list with the oov token into one large
		# list. Start enumerating through that list to assign values to
		# tokens. Note that 0 should always be the value paired with
		# the oov_token.
		self.tokens = [oov_token] + vocabulary
		self.map = {}
		for idx, token in enumerate(self.tokens):
			if invert:
				self.map[idx] = token
			else:
				self.map[token] = idx


	def __call__(self, key):
		# Check that the key is either a string or a value.
		if not isinstance(key, str) and not isinstance(key, int) and not tf.is_tensor(key):
			raise TypeError("key argument must be a string or int")

		if tf.is_tensor(key):
			if key.dtype == "string":
				key = str(key)
			else:
				key = int(key)

		# Handle cases where the key is not present in the map (forward
		# to the OOV token key/value).
		if key not in self.map and isinstance(key, str):
			return tf.constant(
				self.map[self.oov_token], dtype="int64"
			)
		elif key not in self.map and isinstance(key, int):
			return tf.constant(
				self.map[0], dtype="string"
			)

		return tf.constant(
			self.map[key], 
			dtype="int64" if isinstance(key, str) else "string"
		)


	def get_vocabulary(self):
		#return self.tokens
		return list(self.map.keys())


	def vocabulary_size(self):
		#return len(self.tokens)
		return len(self.map.keys())