# Conditional Drums Generation using Compound Word Representations

## Description

A seq2seq architecture where a BiLSTM Encoder receives information about the conditioning parameters (i.e., accompanying tracks and musical attributes), while a Transformer-based Decoder with relative global attention produces the generated drum sequences. For further details please read and cite our paper:

##### Makris D., Guo Z, Kaliakatsos-Papakostas N., Herremans D., “Conditional Drums Generation using Compound Word Representations” to appear EvoMUSART, 2022.

## Prerequisites

-Tensorflow 2.x  <br />
-music21 6.x or lower <br />
-pretty_midi <br />
-pypianoroll <br />
-numpy <br />
-sklearn <br />

## Usage

#### 1. Pre-processing

-pre_process.py: Create CP representations using your own MIDI data. However, we offer the pre-processed dataset used in the paper (rar inside the folder). In that case you do not need to run this file. <br />
-post_process.py: Convert the CP data to one-hot encoding streams ready for training.

#### 2. Training

-model_train.py: Train the model using the hyper-parameters reported in the paper.

#### 3. Inference

-gen_drums.py: Generate conditional drums with your own data. The input MIDI files (./midi_in/) must have two tracks (1st Guitar, 2nd Bass) and with a maximum length of 16 bars. We also offer pre-trained model weights to avoid training [here.](https://drive.google.com/file/d/1gJVrlsukqomC4_U7w1GxLrW-uBxsukhh/view?usp=sharing) Place them inside ./aux_files/checkpoints folder.

#### 4. Samples

Random samples from the experimental setup.

## Reference

If you use this library, please cite the following work:

##### Makris D., Guo Z, Kaliakatsos-Papakostas N., Herremans D., “Conditional Drums Generation using Compound Word Representations” to appear EvoMUSART, 2022.
