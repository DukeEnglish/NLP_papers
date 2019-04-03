# Papers for NLP

 [![](https://jaywcjlove.github.io/sb/license/mit.svg)](#)

NLP is a popular field right now and people are trying to step in this amazing world. I collect some materials here including conferences, papers and typical books. 

Usually, NLP (Natural Lanugage Processing) contains many fields and we view NLP as an interdisciplinary field[2]. So, what is NLP? I copied some things from the anlp slides used in The University of Edinburgh:

| Core technologies           | Applications                     |
| --------------------------- | -------------------------------- |
| • Morphological analysis    | • Machine Translation            |
| • Part-of-speech tagging    | • Information Retrieval          |
| • Syntactic parsing         | • Question Answering             |
| • Named-entity recognition  | • Dialogue Systems               |
| • Coreference resolution    | • Information Extraction         |
| • Word sense disambiguation | • Summarization                  |
| • Textual entailment        | • Sentiment Analysis             |
| • Vector space              | • Natural Language Understanding |
| ...                         | ...                              |





## Conferences in NLP

- [ACL](https://en.wikipedia.org/wiki/Association_for_Computational_Linguistics): The **Association for Computational Linguistics** (**ACL**) is the international scientific and professional society for people working on problems involving [natural language and computation](https://en.wikipedia.org/wiki/Natural_language_and_computation). An annual meeting is held each summer in locations where significant [computational linguistics](https://en.wikipedia.org/wiki/Computational_linguistics) research is carried out. It was founded in 1962, originally named the **Association for Machine Translation and Computational Linguistics** (**AMTCL**). It became the ACL in 1968. Also, here is an interesting [website](https://www.aclweb.org/anthology/) you should have a look. 

  The ACL has a European ([EACL](https://en.wikipedia.org/w/index.php?title=European_Chapter_of_the_Association_for_Computational_Linguistics&action=edit&redlink=1))[[2\]](https://en.wikipedia.org/wiki/Association_for_Computational_Linguistics#cite_note-2) and a North American ([NAACL](https://en.wikipedia.org/wiki/North_American_Chapter_of_the_Association_for_Computational_Linguistics)) chapter.

- NIPS

- AAAI

- COLING

- SIGIR

- [NAACL](https://en.wikipedia.org/wiki/North_American_Chapter_of_the_Association_for_Computational_Linguistics): North American chapter of ACL.

- [EACL](https://en.wikipedia.org/w/index.php?title=European_Chapter_of_the_Association_for_Computational_Linguistics&action=edit&redlink=1): European chapter of ACL.

- [EMNLP](https://en.wikipedia.org/wiki/Empirical_Methods_in_Natural_Language_Processing): **Empirical Methods in Natural Language Processing** or **EMNLP** is a leading conference in the area of [Natural Language Processing](https://en.wikipedia.org/wiki/Natural_Language_Processing). EMNLP is organized by the [ACL](https://en.wikipedia.org/wiki/Association_for_Computational_Linguistics) special interest group on linguistic data (SIGDAT).

- [CoNLL](http://www.signll.org/conll): CoNLL is a top-tier conference, yearly organized by [SIGNLL](http://www.signll.org/) (ACL's Special Interest Group on Natural Language Learning). This year, CoNLL will be colocated with [EMNLP 2019](https://www.emnlp-ijcnlp2019.org/) in Hong Kong. 

Journals：

-  [Computational Linguistics](http://www.transacl.org/): Transactions of ACL. It is the primary forum for research on computational linguistics and [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing). Since 1988, the journal has been published for the ACL by [MIT Press](https://en.wikipedia.org/wiki/MIT_Press). 

中文：

- CCL：
- 全国青年计算语言学研讨会（YCCL）

- 全国信息检索学术会议（CCIR）

- 全国机器翻译研讨会（CWMT）

## Papers in NLP

I will introduce applications&core techs and DL respectively because some methods in App&Core need DL background. I also offer paper review and code for each paper based on Pytorch. (I give up on Tensorflow right now… @.@). Note: Because I don't really know each field below, I didn't give my opinion for some of items.

Applications & core techs:

| Item                           | Papers                                                       | NOTE |
| ------------------------------ | ------------------------------------------------------------ | ---- |
| Machine Translation            | - [Sequence to Sequence Learning with Neural Networks. (Sutskever, Vinyals, Le)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) <br />- [Neural Machine Translation by Jointly Learning to Align and Translate. (Bahdanau, Cho, Bengio)](https://arxiv.org/pdf/1409.0473.pdf)<br /><br />- [Six Challenges for Neural Machine Translation. (Koehn and Knowles)](http://www.aclweb.org/anthology/W17-3204) <br />[What does Attention in Neural Machine Translation Pay Attention to? (Ghader and Monz)](https://arxiv.org/pdf/1710.03348.pdf)<br /><br /><br />[On Using Very Large Target Vocabulary for Neural Machine Translation. (Jean, Cho, Memisevic, Bengio)](http://www.aclweb.org/anthology/P15-1001) <br />[Neural Machine Translation of Rare Words with Subword Units. (Sennrich, Haddow, Birch)](http://www.aclweb.org/anthology/P16-1162.pdf)<br /><br /><br />[RECURRENT NEURAL NETWORK REGULARIZATION](https://arxiv.org/pdf/1409.2329.pdf)<br /><br /><br />[Effective Approaches to Attention-based Neural Machine Translation](https://www.aclweb.org/anthology/D15-1166)<br /><br />[NEURAL MACHINE TRANSLATION<br/>BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf) |      |
| Natural Language Understanding |                                                              |      |
| Question Answering             |                                                              |      |
| Information Retrieval          |                                                              |      |
| Dialogue Systems               |                                                              |      |
| Information Extraction         |                                                              |      |
| Summarization                  |                                                              |      |
| Text classification            |                                                              |      |
| Sentiment Analysis             |                                                              |      |
|                                |                                                              |      |
| Vector space                   |                                                              |      |
| Textual entailment             |                                                              |      |
| Word sense disambiguation      |                                                              |      |
| Coreference resolution         |                                                              |      |
| Named-entity recognition       |                                                              |      |
| Morphological analysis         |                                                              |      |
| Part-of-speech tagging         |                                                              |      |
| Syntactic parsing              |                                                              |      |



Deep Learning for NLP:

Some techs are developed for generaly NLP problem.

| Item                              | Papers                                                       |      |
| --------------------------------- | ------------------------------------------------------------ | ---- |
| Sentence represent                |                                                              |      |
| Basic Embedding Model             |                                                              |      |
| CNN(Convolutional Neural Network) |                                                              |      |
| RNN(Recurrent Neural Network)     |                                                              |      |
| Attention Mechanism               |                                                              |      |
| Model based on Transformer        | - [Attention Is All You Need(2017)](https://arxiv.org/abs/1810.04805), [Transformer_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer_Torch.ipynb), [Transformer(Greedy_decoder)_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer(Greedy_decoder)_Torch.ipynb),<br />- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805), [BERT_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT_Torch.ipynb), |      |



## Books in NLP

It should be hard to read papers directly without background. I recommend people to read a few of books below before following newest papers (I stole them from Internet and most of them are from the course website in some Universities ).

1. Dan Jurafsky and James H. Martin. Speech and Language Processing (3rd ed. draft)

2. Jacob Eisenstein. Natural Language Processing

3. Yoav Goldberg. A Primer on Neural Network Models for Natural Language Processing
4. Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning
5. NMT：<https://arxiv.org/pdf/1709.07809.pdf> (This is an addition for Statistic Machine Translation)
6. [Christopher D. Manning](https://link.zhihu.com/?target=http%3A//nlp.stanford.edu/~manning/), [Prabhakar Raghavan](https://link.zhihu.com/?target=http%3A//theory.stanford.edu/~pragh/), and [Hinrich Schütze](https://link.zhihu.com/?target=http%3A//www-csli.stanford.edu/~hinrich). 2008.[Introduction to Information Retrieval](https://link.zhihu.com/?target=http%3A//nlp.stanford.edu/IR-book/). Cambridge University Press.
7. [Linguistic Fundamentals for Natural Language Processing](http://www.morganclaypool.com/doi/abs/10.2200/S00493ED1V01Y201303HLT020) by Emily Bender. 

Dr Zhiyuan Liu recommended these books for Chinese student.

NLP:

1. 统计自然语言处理
2. 信息检索导论

ML:

1. 统计机器学习-李航

2. 机器学习-周志华

DL：

1. Michael A. Nielsen. Neural Networks and Deep Learning

2. Eugene Charniak. Introduction to Deep Learning

AI：

1. 人工智能：一种现代的方法

社会计算：

2. 网络、群体与市场：揭示高度互联世界的行为原理与效应机制

必读1.4部分



## Tool/Lib/framework in NLP

"If I have seen further,it is by standing on the shoulders of giants." 

- [NLTK](http://www.nltk.org/): 
- [genism](https://radimrehurek.com/gensim/about.html)：
- [allennlp](https://allenai.github.io/allennlp-docs/)：
- LTP
- jieba
- spaCy
- coreNLP
- StanfordNLP
- TextBolb



## People in NLP

I do think follow people who are excellent in this field is a good idea. Many people are famous and excellent and I just post a few I know and I do learn something about NLP/ML from them. There are also others excellent I don't know yet. Hopefully I could know them soon~~~

[Adam lopez](http://alopez.github.io/collaborators/): My advisor in Edinburgh, whose attitude to his students and study touched me deeply.

[Nikolay Bogoychev](https://nbogoychev.com/): OMG, he is quite good at language. Thanks to his taught when I was in Edinburgh.

Zhiyuan Liu（刘知远）: 
[Danqi chen](https://cs.stanford.edu/people/danqi/): emmm. She looks beautiful. Her dissertation is quite excellent and popular. 



## Labs in NLP

[ILCC](http://web.inf.ed.ac.uk/ilcc): 



## References

1. https://github.com/daicoolb/RecommenderSystem-Paper: This is a repository developed by my schoolmate Jie Liu in HUNAN University about things (paper, tool, framework) for RecommenderSystem. (He is an awesome boy BTW)
2. http://alopez.github.io: This is the personal page of my advisor Adam Lopez in The University of Edinburgh. I have learned a lot from him, especially from his opinion about [Self-funded students](http://alopez.github.io/join/#self-funded-students).
3. https://github.com/llhthinker/NLP-Papers#distributed-word-representations: 
4. https://github.com/graykode/nlp-tutorial:







