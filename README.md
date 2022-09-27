# Question Answering and Question Generation using 🤗Transformers


## idT5 for Question Generation and Question Answering

[idT5](https://huggingface.co/muchad/idt5) (Indonesian version of [mT5](https://huggingface.co/google/mt5-base)) is fine-tuned on [translated SQuAD v2.0](https://github.com/Wikidepia/indonesian_datasets/tree/master/question-answering/squad) for **Question Generation and Question Answering** tasks.

# Live Demo
**Question Generation:** [ai.muchad.com/qg](https://ai.muchad.com/qg/)
**Question Answering:** [t.me/caritahubot](https://t.me/caritahubot)

## Requirements
```
!pip install transformers==4.4.2
!pip install sentencepiece==0.1.95
!git clone https://github.com/muchad/qaqg.git
%cd qaqg/
```

## Usage 🚀
### Question Answering

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muchad/qaqg/blob/main/idT5_Question_Answering.ipynb)

```
from pipeline_qa import pipeline #pipeline_qa.py script in the cloned repo
qa = pipeline()

#sample
qa({"context":"Raja Purnawarman mulai memerintah Kerajaan Tarumanegara pada tahun 395 M.","question":"Siapa pemimpin Kerajaan Tarumanegara?"})

#output
=> Raja Purnawarman
```
