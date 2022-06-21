# @Author: Darryl Estrada
# @Date: 9/06/2022
# @Version: 1.0
# @Description: This is the main file for the training of a spacy model



import os
from os import listdir
from os.path import isfile, join
import random
import numpy as np
import pandas as pd
import thinc
import spacy
from spacy.util import filter_spans
nlp=spacy.load('es_dep_news_trf')
import re
import time
import sys
import argparse
from tqdm import tqdm
from spacy.tokens import DocBin
import subprocess
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Load the training data
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--data', type=str, default='data/', help='Path to training, dev and test data')
parser.add_argument('--output', type=str, default= 'output/', help='Path to save the model')
parser.add_argument('--gpu', type=int, default=None, help='GPU id to use, ex = 0,1,2,3')
parser.add_argument('--model', type=str, default='', help='Model to use')
parser.add_argument('--train', type=str2bool, default= True , help='True or False, default True, if False, prediction will be done')
parser.add_argument('--prediction_output', type=str, default="predictions/", help='Path to save the predictions')
args = parser.parse_args()
print(args)

def get_unique_file_names(mypath):
    onlyfiles = [os.path.splitext(f)[0] for f in listdir(mypath) if isfile(join(mypath, f))]
    return list(set(onlyfiles))

# Python program to sort a list of
# tuples by the second Item using sorted() 
  
# Function to sort the list by second item of tuple
def Sort_Tuple(tup): 
  
    # reverse = None (Sorts in Ascending order) 
    # key is set to sort using first element of 
    # sublist lambda has been used 
    Sorted = sorted(tup, key = lambda x: x[0])
    
    last_start = 0
    last_end = 0
    new_tup = []
    for t in Sorted:
        try:
            if t[0] not in range(last_start, last_end) and t[1] not in range(last_start, last_end) and (t[1]-t[0]) < 35:
                last_start = t[0]
                last_end = t[1]
                new_tup.append((t[0],t[1],t[2]))
        except:
            print("Sort_Tuple error")
    return(new_tup) 

def create_training_data():
    ### Get unique File
    files = get_unique_file_names(args.data+"/train")
    TRAIN_DATA = []
    print("total documents for training "+ str(len(files)))
    for file in files:
        try:
            with open(args.data+"/train/"+file+".txt",'r', encoding="UTF-8") as doc:
                text = doc.read()
            with open(args.data+"/train/"+file+".ann",'r', encoding="UTF-8") as doc2:
                ann = doc2.readlines()     
            ann1 = {}
            ann_array = []
            for a in ann:
                annotations = a.split("\t")
                if (annotations[0][0] == 'T' and annotations[0][1].isdigit()):
                    an = annotations[1].split(" ")
                    du = (int(an[1]),int(an[2]),an[0])
                    ann_array.append(du)
            if(len(ann_array) > 0):
                ann1['entities'] = Sort_Tuple(ann_array)
                dup_final = (text,ann1)
                TRAIN_DATA.append(dup_final)
            else:
                print("No annotations for file "+file)
        except Exception as e: print(e,file)

    
    ### Get unique File
    files_env = get_unique_file_names(args.data+"/valid")
    VALID_DATA = []
    print("total documents for validation "+ str(len(files_env)))
    for file in files_env:
        try:
            with open(args.data+"/valid/"+file+".txt",'r', encoding="UTF-8") as doc:
                text = doc.read()
            with open(args.data+"/valid/"+file+".ann",'r', encoding="UTF-8") as doc2:
                ann = doc2.readlines()
                
            ann1 = {}
            ann_array = []
            for a in ann:
                annotations = a.split("\t")
                if (annotations[0][0] == 'T' and annotations[0][1].isdigit()):
                    an = annotations[1].split(" ")
                    du = (int(an[1]),int(an[2]),an[0])
                    ann_array.append(du)
            if(len(ann_array) > 0):
                ann1['entities'] = Sort_Tuple(ann_array)
                dup_final = (text,ann1)
                VALID_DATA.append(dup_final)
        except Exception as e: print(e,file)
    return TRAIN_DATA, VALID_DATA


def create_spacy_file(data,type:str):
    # Create a DocBin to store the training data
    db = DocBin() # create a DocBin object
    for text, annot in tqdm(data): # data in previous format
        doc = nlp.make_doc(text) # create doc object from text
        ents = []
        for start, end, label in annot["entities"]: # add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                ents.append(span)
        doc.spans["sc"] = ents # label the text with the ents
        db.add(doc)
    save_file = str(args.data)+"/"+type+".spacy"
    db.to_disk(save_file) # save the docbin object

def create_init_conf_file():
    txt = """# This is an auto-generated partial config. To use it with 'spacy train'
# you can run spacy init fill-config to auto-fill all default settings:
# python -m spacy init fill-config ./base_config.cfg ./config.cfg
[paths]
train = null
dev = null
vectors = null
[system]
gpu_allocator = "pytorch"

[nlp]
lang = "en"
pipeline = ["transformer","spancat"]
batch_size = 128

[components]

[components.transformer]
factory = "transformer"

[components.transformer.model]
@architectures = "spacy-transformers.TransformerModel.v3"
name = "roberta-base"
tokenizer_config = {"use_fast": true}

[components.transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96

[components.spancat]
factory = "spancat"
max_positive = null
scorer = {"@scorers":"spacy.spancat_scorer.v1"}
spans_key = "sc"
threshold = 0.5

[components.spancat.model]
@architectures = "spacy.SpanCategorizer.v1"

[components.spancat.model.reducer]
@layers = "spacy.mean_max_reducer.v1"
hidden_size = 128

[components.spancat.model.scorer]
@layers = "spacy.LinearLogistic.v1"
nO = null
nI = null

[components.spancat.model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0

[components.spancat.model.tok2vec.pooling]
@layers = "reduce_mean.v1"

[components.spancat.suggester]
@misc = "spacy.ngram_suggester.v1"
sizes = [1,2,3]

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0

[training]
accumulate_gradient = 3
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"

[training.optimizer]
@optimizers = "Adam.v1"

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250
total_steps = 20000
initial_rate = 5e-5

[training.batcher]
@batchers = "spacy.batch_by_padded.v1"
discard_oversize = true
size = 2000
buffer = 256

[initialize]
vectors = ${paths.vectors}"""
    with open(args.data+"/base_config.cfg",'w', encoding="UTF-8") as doc:
        doc.write(txt)

    bashCommand = "python -m spacy init fill-config "+args.data+"/base_config.cfg "+args.data+"/config.cfg"
    print(bashCommand)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output,error)




if __name__ == "__main__":
   
    if args.train == False and args.model == '':
        print("You need to provide a model to use for prediction")
        exit()
    
    if args.train == True:
        train, valid = create_training_data()
        create_spacy_file(train, "train")
        create_spacy_file(valid, "dev")
        create_init_conf_file()
        os.makedirs(args.output, exist_ok=True)
        gpu = args.gpu

        if gpu != None:
            spacy.require_gpu(gpu_id=gpu)
            bashCommand = "python -m spacy train "+args.data+"/config.cfg --output "+args.output+" --paths.train "+args.data+"/train.spacy --paths.dev "+args.data+"/dev.spacy --gpu-id "+str(args.gpu)
        else:
            bashCommand = "python -m spacy train "+args.data+"/config.cfg --output "+args.output+" --paths.train "+args.data+"/train.spacy --paths.dev "+args.data+"/dev.spacy"
        print(bashCommand)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output,error)

    
    
    