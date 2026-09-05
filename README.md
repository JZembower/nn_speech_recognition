# nn_speech_recognition

Frame-level **phoneme labelling** with a multilayer perceptron (MLP).

Given log-mel spectrogram frames from speech utterances (LibriSpeech), the model predicts one of 40 phoneme labels for each frame, using optional surrounding context. Training data includes spectrograms and labels; the test set has spectrograms only.

## Objectives

- Set up, train, and optimize an MLP for this medium-scale classification task, including data loading.
- Explore architectures and hyperparameters, and search that space systematically.
- Stage the work: start simple, iterate on data subsets, track validation loss/accuracy, then scale promising setups to the full dataset.
- Implement in PyTorch with attention to memory, precision, and training/inference efficiency.
