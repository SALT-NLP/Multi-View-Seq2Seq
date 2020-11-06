# Multi-View-Seq2Seq
This repo contains codes for the following paper: 

*Jiaao Chen, Diyi Yang*: Multi-View Sequence-to-Sequence Models with Conversational Structure for Abstractive Dialogue Summarization,  EMNLP 2020

If you would like to refer to it, please cite the paper mentioned above. 


## Getting Started
These instructions will get you running the codes of Multi-View Conversation Summarization.

### Requirements
* Python 3.6 or higher
* Pytorch >= 1.3.0
* Pandas, Numpy, Pickle
* rouge(https://github.com/pltrdy/rouge)
* Fairseq
* sentence_transformers


### Code Structure
```
|__ data/
        |__ C99.py, C99utils.py --> C99 topic segmentation functions
        |__ Sentence_Embeddings.ipynb --> Jupyter Notebook for getting the embeddings for utterances using SentBert
        |__ Topic_Segment.ipynb --> Jupyter Notebook for getting the topic segments using C99
        |__ Stage_Segment.ipynb --> Jupyter Notebook for getting the stage segments using HMM
        |__ Read_Labels.ipynb --> Jupyter Notebook for getting the formated data for traning/evaluation
        |__ Please download the full data folder from here https://drive.google.com/file/d/1-W42dS74MuFQUKBIru6_yc2Sm7LObc7o/view?usp=sharing

|__fairseq_multi_view/ --> Source codes built on fairseq, containing the multi-view model codes
|__train_sh/
        |__*_data_bin --> Store the binarized files
        |__bpe.sh, binarize.sh --> Pre-process the data for fairseq training
        |__train_multi_view.sh, train_single_view.sh --> Train the models
```

### Install the multi-view-fairseq

```
cd fairseq_multi_view

pip install --editable ./
```


### Downloading the data
Please download the dataset and put them in the data folder [here](https://drive.google.com/file/d/1-W42dS74MuFQUKBIru6_yc2Sm7LObc7o/view?usp=sharing)

### Pre-processing the data

The data folder you download from the above link already contains all the pre-processed files from SamSUM corpus.

#### Segment conversations

For your own data, first go through `Sentence_Embeddings.ipynb` to store all the embeddings of utterances in pickle files. Then using `Topic_Segment.ipynb` and `Stage_Segment.ipynb` to read the utterance representations and segment the conversations. You will generate the `*_label.pkl`, which contains the segment id for each utterance in conversations. Finally, using `Read_Labels.ipynb` to generate segmented data `*.source` and `*.target` for fairseq framework.

#### BPE preprocess:

```
cd train_sh

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

./bpe.sh
```

#### Binarize dataset:
```
cd train_sh

./binarize.sh
```

### Download the pre-trained BART
Please download the pre-trained model from [here](https://github.com/pytorch/fairseq/tree/master/examples/bart). And modify the BART_PATH in `./train_single_view.sh` or `./train_multi_view.sh`.
 
### Training models

These section contains instructions for training the conversation summarizationmodels.

The trained multi-view summarization models used in the paper can be downloaded [here](https://drive.google.com/file/d/1Rhzxk1B7oaKi85Gsxr_8WcqTRx23HO-y/view?usp=sharing).
The generated summaries on test set is in the data folder.

Note that during training, after every epoch, it will automatically evaluate on the val and test set (you might need to change the dataset path in `./fairseq_multi_view/fairseq_cli/train.py` for single_view training). The best model is selected based on lower loss on val set. Also, the training is performed on one P100 GPU (or other GPU with memory >= 16G). After 6 or 7 epoches, it will get the best model and you could stop further training.

#### Training Single-View model
Please run `./train_single_view.sh` to train the single-view models. Note that you might need to modify the data folder name.


#### Training Multi-View model
Please run `./train_multi_view.sh` to train the Multi-view model, where it combines topic view and stage view. If you are going to combine different views, please modify the corresponding data folder name as well.

### Evaluating models

An example jupyter notebook (`Eval_Sum.ipynb`) is provided for evaluating the model on test set.





