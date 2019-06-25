# Twitter Sentiment Analysis

#### Training the models

Run `init.sh` before anything, in order to download the required data and dependencies. The full list of trainable models is:
  - `simple-rnn`
  - `cnn1layer`
  - `cnn-multiple-kernels`
  - `birnn`


  - `transfer-deeprnnn`
  - `transfer-kernels`


  - `elmo`
  - `elmomultilstm`

Furthermore, we trained a model using Bert encoding. Instructions for training and using its predictions can be found in the last section.

In order to train one of the models use the `main.py` script:
```bash
usage: main.py [-h] [--eval] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
               [--load LOAD] [--transfer TRANSFER] [--ensemble ENSEMBLE]
               model_name

positional arguments:
  model_name            Model name (e.g. cnnlstm)

optional arguments:
  -h, --help            show this help message and exit
  --eval                Specify this option to not train but only eval using
                        the last saved model or a specific checkpoint with
                        --load
  --batch_size BATCH_SIZE
                        Batch size to use during training.
  --epochs EPOCHS       Number of epochs to train for
  --load LOAD           Specify some checkpoint to load. Specify the .hdf5
                        file without .data or .index afterwards
  --transfer TRANSFER   Use the loaded model for transfer learning.
  --ensemble ENSEMBLE   Ensemble size to use
```

For example, the model `birnn` has been trained as follows:
```bash
python3 main.py --epochs 5 birnn
```

Note that the models `trasnfer-deeprnn` and `transfer-kernels` use parts of a trained `birnn` model. To train such models, one can run a command similar to the following one:
```bash
python3 main.py --epochs 1 --load <path-to-base-model> --transfer birnn transfer-deeprnn
```

#### Testing the models

To test a model, one can give either a `.hdf5` or `.bin` file as argument to the `load` argument of `main.py`. Below, you can find an example:
```bash
python3 main.py --eval --load models/birnn.bin birnn
```

The command above will generate 2 files in the `preds` folder:
 - `<modelname>-<id>.csv` - the CSV file of predictions
 - `<modelname>-<id>-debug.csv` - a CSV file with numbers between 0 and 1, representing the certainty of the predictions

#### Bert

To be able to use the Bert model, download the following pretrined embedding:

```bash
mkdir -p models
pushd models
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
unzip uncased_L-12_H-768_A-12.zip
unzip uncased_L-24_H-1024_A-16.zip
popd > /dev/null
```

This module requires Tensorflow 2.0, so download the dependency before training the model. Training can be done via the command
```bash
python3 model_bert.py
```

In the end, predictions can be made by running:
```bash
python3 model_bert.py --predict --weights <path-to-trained-model>
```
