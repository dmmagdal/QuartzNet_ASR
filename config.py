# config.py
# Simple config class used for initializing a QuartzNet model.
# Note:
# There are 4 Convolutional sections C, 5 Blocks B, each block B
# with 5 Time Separable 1D Convolutional sections S. The blocks B are
# repeated R times.
# Model name = (B x R) X S
# B = 5, R = {1, 2, 3} (aka block_repeat), S = 5 (module_repeat)
# Tensorflow 2.4
# Python 3.7
# Windows/MacOS/Linux


import os
import json


class Config:
	def __init__(self, block_repeat=3, module_repeat=5):
		# Model | block_repeat | module_repeat
		# 5x5           1              5
		# 10x5          2              5
		# 15x5          3              5
		self.block_repeat = block_repeat
		self.module_repeat = module_repeat


	def save_config(self, path_dir):
		if not os.path.exists(path_dir):
			os.makedirs(path_dir, exist_ok=True)
		with open(os.path.join(path_dir, "config.json"), "w+") as f:
			data = {
				"block_repeat": self.block_repeat,
				"module_repeat": self.module_repeat,
			}
			json.dump(data, f, indent=4)


	def load_config(self, path_dir):
		config_file = os.path.join(path_dir, "config.json")
		if not os.path.exists(path_dir):
			print(f"Error: Could not locate path: {path_dir}")
		elif not os.path.exists(config_file):
			print(f"Error: Could not locate config file: {config_file}")
		with open(os.path.join(path_dir, "config.json"), "w+") as f:
			data = json.load(f)
		self.block_repeat = data["block_repeat"]
		self.module_repeat = data["module_repeat"]