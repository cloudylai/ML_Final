# ML_Final
Machine Learning Final Project

## Introduction
This repository is a duplication of the final team project, Delta Chinese QA, of NTU Machine Learning Course ([website](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17_2.html)) in Fall, 2017. This project is worked by Chiu-Te Wang, Chao-Chung Wu, and Chih-Te Lai. The objective of this project is to solve a Chinese question answering task; that is, given a Chinse context and a question, one aims to train a machine which indicates the positions in the context corresponding to the answer of the question. We use the training data and testing data provided by the course. A final report describing multiple experiments and comparsions of different preprocessing and models is contained in results/ directory.

## Method
Our general framework can be seen below. A embedding layer is used to obtain proper representation of question and context in both word-level and character-level. Encoder layers are used to extract the semantic . Specifically, in this project, we utilize two popular models in SQuAD ([link](https://rajpurkar.github.io/SQuAD-explorer/)): BiDAF ([link](https://arxiv.org/abs/1611.01603)) and R-NET ([link](https://www.microsoft.com/en-us/research/publication/mrc/)) with some adjustments to improve their performance in our task. 

![image1](https://github.com/cloudylai/ML_Final/blob/master/images/framework_1.png)  

## Result


## Reference
### Paper:  
1. Natural Language Computing Group, Microsoft Research Asia. R-NET: Machine Reading Comprehension with Self-Machine Networks. [link](https://www.microsoft.com/en-us/research/publication/mrc/)  
2. Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi. Bidirectional Attention Flow for Machine Comprehension. [link](https://arxiv.org/abs/1611.01603)  
### webiste:  
1. SQuAD: [link](https://rajpurkar.github.io/SQuAD-explorer/)
### github:  
1. NLPLearn R-net: [link](https://github.com/NLPLearn/R-net)  
2. allenai BiDAF: [link](https://github.com/allenai/bi-att-flow)  

