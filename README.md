# Text based depression analysis

Source code for the paper (Text-based Depression Detection: What Triggers An Alert)[]


# Prequisites

The required python packages can be found in `requirements.txt`.

```
tqdm==4.24.0
allennlp==0.8.2
tableprint==0.8.0
tabulate==0.8.2
fire==0.1.3
nltk==3.3
kaldi_io==0.9.1
scipy==1.2.1
torchnet==0.0.4
pandas==0.24.1
numpy==1.16.2
bert_serving_client==1.8.3
imbalanced_learn==0.4.3
torch==0.4.1.post2
gensim==3.7.1
imblearn==0.0
scikit_learn==0.20.3
PyYAML==5.1
```


## Dataset 

The dataset is freely available for research purposes and can be found [here](http://dcapswoz.ict.usc.edu/).


## Preprocessing

The preprocessing feature extraction scripts can be found in `data_preprocess/`.

We exclusively use [kaldi](http://kaldi-asr.org/) as our data processing pipeline ( except for extraction ). 

Word2Vec features require manual training. For this purpose, one should create some `all_transcripts.csv` file with the same format as all other `*TRANSCRIPTS.csv` files.

The easies way to do that is just use bash or pandas (python) to merge all files into one, removing all headers except the first one e.g.,

```bash
cd labels_processed;
python -c from glob import glob as g; import pandas as pd; pd.concat([pd.read_csv(f,sep='\t') for f in g('*TRANS*.csv')]).to_csv('all_transcripts.csv', index=False, sep='\t')
```

Then just run the `extract_text_gensim.py` script, which extracts `100dim` features into the `features/text` folder.

ELMo features can be rather directly downloaded and pretrained given one installed `allennlp`. 
However, BERT embeddings require the download of a [pretrained BERT](https://github.com/google-research/bert).
In order to have easy access to BERT embeddings we use [bert serving client](https://pypi.org/project/bert-serving-client/). Here we prepared a simple script `start_bert_server.sh` in order to start the serving client. We recommend the use of screen or tmux.



In order to extract the features, prepare a folder named `labels` and softlink the `train_split_Depression_AVEC2017.csv` and dev files into it.

Further prepare the transcripts for feature extraction by directly copying all transcripts into a dir called `labels_processed`.
In other cases pass `--transcriptdir OUTDIRNAME` to any of the extraction scripts.


# Running the code

The main script of this repo is `run.py`.

The code is centered around the config files placed at `config/`. Each parameter in these files can be modified for each run using google-fire e.g., if one opts to run a different model, just pass `--model GRU`. 

`run.py` the following options ( ran as `python run.py OPTION`):
* `train`: Trains a model given a config file (default is `config/text_lstm_deep.yaml`)
* `stats`: Prints the evaluation results on the development set
* `trainstats`: Convenience function to run train and evaluate in one.
* `search`: Parameter search for learning rate, momentum and nesterov (SGD)
* `searchadam`: Learning rate search for adam optimizer
* `ex`: Extracts features from a given network (not finished),
* `fwd`: Debugging function to forward some features through a network
* `fuse`: Fusion of two models or more. Fusion is done by averaging each output.


