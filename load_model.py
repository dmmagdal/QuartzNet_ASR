# load_model.py
# Small test program that tests initializing the QuartzNet model.
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import string
from config import Config
from quartznet import QuartzNet, StringMap


def main():
	# Initialize variables like c_in and c_out. With QuartzNet for ASR
	# (automatic speech recognition), c_in is the fft_length // 2 + 1
	# and the c_out is the character vocabulary.
	n_fft = 1024
	c_in = n_fft // 2 + 1
	c_out = len(string.ascii_lowercase + "?.!'" )

	# Initialize build shape (batch_size, time_steps, c_in).
	input_shape = (None, None, c_in)

	# Initialize a QuartzNet 5x5 and give the summary.
	cfg1 = Config(1)
	quartz_5x5 = QuartzNet(c_in, c_out, cfg1, name="quartznet_5x5")
	quartz_5x5.build(input_shape=input_shape)
	quartz_5x5.summary()

	# Initialize a QuartzNet 10x5 and give the summary.
	cfg2 = Config(2)
	quartz_10x5 = QuartzNet(c_in, c_out, cfg2, name="quartznet_10x5")
	quartz_10x5.build(input_shape=input_shape)
	quartz_10x5.summary()

	# Initialize a QuartzNet 5x5 and give the summary.
	cfg3 = Config(3)
	quartz_15x5 = QuartzNet(c_in, c_out, cfg3, name="quartznet_15x5")
	quartz_15x5.build(input_shape=input_shape)
	quartz_15x5.summary()

	# Initialize characters list and use that to test out the custom
	# StringMap class that covers the functions necessary for this ASR
	# task. If using Tensorflow >= 2.6.0, use 
	# tf.keras.layers.StringLookup as seen in the original Automatic
	# Speech Recognition using CTC example on Keras.
	chars = list(string.ascii_lowercase + "?!' ")
	char_to_num = StringMap(vocabulary=chars, oov_token="")
	print(f"Char -> Num vocab: {char_to_num.get_vocabulary()}")
	print(f"Char -> Num vocab size: {char_to_num.vocabulary_size()}")
	print(f"Char -> Num (a) ")
	print(char_to_num("a"))
	print(f"Char -> Num (b2) which is OOV ")
	print(char_to_num("b2"))
	num_to_char = StringMap(
		vocabulary=char_to_num.get_vocabulary(), oov_token="", 
		invert=True
	)
	print(f"Num -> Char vocab: {num_to_char.get_vocabulary()}")
	print(f"Num -> Char vocab size: {num_to_char.vocabulary_size()}")
	print(f"Num -> Char (4) ")
	print(num_to_char(4))
	print(f"Num -> Char (99) which is OOV ")
	print(num_to_char(99))
	num_to_char = StringMap(
		vocabulary=char_to_num.get_vocabulary(), oov_token="", 
		invert=True
	)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()