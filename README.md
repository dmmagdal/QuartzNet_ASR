# QuartzNet ASR

### Description

Implement Nvidia's QuartzNet neural net for the task of Automatic Speech Recognition (ASR) in Tensorflow 2. QuartzNet comes from the Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions paper and was trained with CTC loss on the LibriSpeech dataset to achieve state-of-the-art (SOTA) accuracy with a Word Error Rate (WER) in the range of 4.19 to 10.98%. QuartzNet can be found as part of Nvidia's NeMo repository on Github however, this implementation was based off of Jaco-Assistant GitLab repository and is trained on the LJSpeech dataset from Keith Ito. Training setup is taken from the Automatic Speech Recognition using CTC example from the Keras examples page.

### Scripts

 > load_model.py
 >> Initializes 3 QuartzNet models (5x5, 10x5, 15x5) as well as testing the custom StringMap class in quartznet.py used as a functional replacement for tf.keras.layers.StringLookup for those running versions Tensorflow below 2.6.0.

 > asr_ctc_quartznet.py
 >> A spinoff of the ASR using CTC Keras example that replaces the DeepSpeech2 model with QuartzNet 15x5 and trains on the LJSpeech dataset.

 > Dockerfile
 >> Dockerfile for running asr_ctc_quartznet.py in a docker container. 

### Sources

 > [Nvidia QuartzNet paper](https://arxiv.org/pdf/1910.10261.pdf)
 > [Nvidia NeMo GitHub](https://github.com/NVIDIA/NeMo/tree/main/nemo)
 > [Jaco-Assistant GitLab](https://gitlab.com/Jaco-Assistant/Scribosermo/)
 > [LibriSpeech download page](https://www.openslr.org/12/)
 > [LJSpeech download page](https://keithito.com/LJ-Speech-Dataset/)
 > [Automatic Speech Recognition using CTC Keras example](https://keras.io/examples/audio/ctc_asr/)