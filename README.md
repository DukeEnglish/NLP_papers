# Papers for NLP

[![](https://jaywcjlove.github.io/sb/license/mit.svg)](#) [![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu) [![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

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

- [NIPS](https://en.wikipedia.org/wiki/Conference_on_Neural_Information_Processing_Systems): The Conference and Workshop on **Neural Information Processing Systems** (abbreviated as **NeurIPS** and formerly **NIPS**) is a machine learning and computational neuroscience conference held every December. The conference is currently a double-track meeting (single-track until 2015) that includes invited talks as well as oral and poster presentations of refereed papers, followed by parallel-track workshops that up to 2013 were held at ski resorts.

- [AAAI](https://en.wikipedia.org/wiki/Association_for_the_Advancement_of_Artificial_Intelligence): The **Association for the Advancement of Artificial Intelligence** (**AAAI**) is an international scientific society devoted to promote research in, and responsible use of, [artificial intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence). AAAI also aims to increase public understanding of artificial intelligence (AI), improve the teaching and training of AI practitioners, and provide guidance for research planners and funders concerning the importance and potential of current AI developments and future directions.

- ICLR: ICLR is the hottest conference these years.

- [SIGIR](https://en.wikipedia.org/wiki/Special_Interest_Group_on_Information_Retrieval): **SIGIR** is the [Association for Computing Machinery](https://en.wikipedia.org/wiki/Association_for_Computing_Machinery)'s **Special Interest Group on Information Retrieval**. The scope of the group's specialty is the theory and application of computers to the acquisition, organization, storage, [retrieval](https://en.wikipedia.org/wiki/Information_retrieval) and distribution of information; emphasis is placed on working with non-numeric information, ranging from natural language to highly structured data bases.

- [NAACL](https://en.wikipedia.org/wiki/North_American_Chapter_of_the_Association_for_Computational_Linguistics): North American chapter of ACL.

- [EACL](https://en.wikipedia.org/w/index.php?title=European_Chapter_of_the_Association_for_Computational_Linguistics&action=edit&redlink=1): European chapter of ACL.

- [EMNLP](https://en.wikipedia.org/wiki/Empirical_Methods_in_Natural_Language_Processing): **Empirical Methods in Natural Language Processing** or **EMNLP** is a leading conference in the area of [Natural Language Processing](https://en.wikipedia.org/wiki/Natural_Language_Processing). EMNLP is organized by the [ACL](https://en.wikipedia.org/wiki/Association_for_Computational_Linguistics) special interest group on linguistic data (SIGDAT).

- [CoNLL](http://www.signll.org/conll): CoNLL is a top-tier conference, yearly organized by [SIGNLL](http://www.signll.org/) (ACL's Special Interest Group on Natural Language Learning). This year, CoNLL will be colocated with [EMNLP 2019](https://www.emnlp-ijcnlp2019.org/) in Hong Kong. 

Journals：

-  [Computational Linguistics](http://www.transacl.org/): Transactions of ACL. It is the primary forum for research on computational linguistics and [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing). Since 1988, the journal has been published for the ACL by [MIT Press](https://en.wikipedia.org/wiki/MIT_Press). 

## Papers in NLP

I will introduce applications&core techs and DL respectively because some methods in App&Core need DL background. I also offer paper review and code for each paper based on Pytorch. (I give up on Tensorflow right now… @.@). Note: Because I don't really know each field below, I didn't give my opinion for some of items. (I wrote reviews for some of them)

#### Applications & core techs:

##### Machine Translation: 

1. [Attention Is All You Need(2017)](https://arxiv.org/abs/1810.04805), [Transformer_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer_Torch.ipynb), [Transformer(Greedy_decoder)_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer(Greedy_decoder)_Torch.ipynb),
2. [Sequence to Sequence Learning with Neural Networks. (Sutskever, Vinyals, Le, 2014)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) , review
3. [Neural Machine Translation by Jointly Learning to Align and Translate. (Bahdanau, Cho, Bengio)](https://arxiv.org/pdf/1409.0473.pdf)
4. [Six Challenges for Neural Machine Translation. (Koehn and Knowles)](http://www.aclweb.org/anthology/W17-3204) 
5. [What does Attention in Neural Machine Translation Pay Attention to? (Ghader and Monz)](https://arxiv.org/pdf/1710.03348.pdf)
6. [On Using Very Large Target Vocabulary for Neural Machine Translation. (Jean, Cho, Memisevic, Bengio)](http://www.aclweb.org/anthology/P15-1001) 
7. [Neural Machine Translation of Rare Words with Subword Units. (Sennrich, Haddow, Birch)](http://www.aclweb.org/anthology/P16-1162.pdf)
8. [RECURRENT NEURAL NETWORK REGULARIZATION](https://arxiv.org/pdf/1409.2329.pdf)
9. [Effective Approaches to Attention-based Neural Machine Translation](https://www.aclweb.org/anthology/D15-1166)
10. [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf)
11. [Guillaume Klein, Yoon Kim, Yuntian Deng, Jean Senellart, and Alexander Rush. 2017. OpenNMT: Open-source toolkit for neural machine translation. pages 67–72.](<https://arxiv.org/pdf/1701.02810.pdf>)
12. Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: a method for automatic evaluation of machine translation. In Association for Computational Linguistics (ACL), pages 311–318.
13. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate. In International Conference on Learning Representations (ICLR).
14. Thang Luong, Hieu Pham, and Christopher D Manning. 2015. Effective approaches to attention-based neural machine translation. In Empirical Methods in Natural Language Processing (EMNLP), pages 1412–1421.
15. Shiqi Shen, Yong Cheng, Zhongjun He, Wei He, Hua Wu, Maosong Sun, and Yang Liu. 2016. Minimum risk training for neural machine translation. In Association for Computational Linguistics (ACL), volume 1, pages 1683–1692.

##### Nutural Language Understanding:

1. [Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation (Ling et al. 2015)](https://aclweb.org/anthology/D15-1176)
2. [From Characters to Words to in Between: Do We Capture Morphology? (Vania and Lopez 2017)](http://aclweb.org/anthology/P17-1184)
3. [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations (Kiperwasser and Goldberg, 2016)](https://aclweb.org/anthology/Q16-1023)
4. [Recurrent neural network grammars (Dyer et al. 2016)](https://arxiv.org/pdf/1602.07776.pdf)
5. [What do RNNGs learn about syntax? (Kuncoro et al. 2017)](http://www.aclweb.org/anthology/E17-1117)
6. [End-to-end Learning of Semantic Role Labeling Using Recurrent Neural Networks (Zhou & Xu, 2015)](http://www.aclweb.org/anthology/P15-1109)
7. [Language to logical form with neural attention (Dong & Lapata, 2016)](http://www.aclweb.org/anthology/P16-1004)
8. [Background: Jurafsky and Martin, Intro to Ch. 28 and section 28.2 (third edition)](https://web.stanford.edu/~jurafsky/slp3/)
9. [A Fully Bayesian Approach to Unsupervised Part-of-Speech Tagging (Goldwater & Griffiths, 2007)](http://www.aclweb.org/anthology/P/P07/P07-1094.pdf)
10. [The Social Impact of Natural Language Processing (Hovy and Spruit, 2016)](http://www.aclweb.org/anthology/P16-2096)
11. [Semantics derived automatically from language corpora contain human-like biases (Caliskan et al., 2017)](http://opus.bath.ac.uk/55288/4/CaliskanEtAl_authors_full.pdf)
12. [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings (Bolukbasi et al., 2016)](http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)
13. Saku Sugawara, Kentaro Inui, Satoshi Sekine, and Akiko Aizawa. 2018. What makes reading comprehension questions easier? In Empirical Methods in Natural Language Processing (EMNLP), pages 4208–4219.
14. Saku Sugawara, Yusuke Kido, Hikaru Yokono, and Akiko Aizawa. 2017. Evaluation metrics for machine reading comprehension: Prerequisite skills and readability. In Association for Computational Linguistics (ACL), volume 1, pages 806–817.
15. Takeshi Onishi, Hai Wang, Mohit Bansal, Kevin Gimpel, and David McAllester. 2016. Who did what: A large-scale person-centered cloze dataset. In Empirical Methods in Natural Language Processing (EMNLP), pages 2230–2235.
16. Karthik Narasimhan and Regina Barzilay. 2015. Machine comprehension with discourse relations. In Association for Computational Linguistics (ACL), volume 1, pages 1253–1262.
17. Alexander Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, and Jason Weston. 2016. Key-value memory networks for directly reading documents. In Empirical Methods in Natural Language Processing (EMNLP), pages 1400–1409.
18. Danqi Chen, Jason Bolton, and Christopher D Manning. 2016. A thorough examination of the CNN/Daily Mail reading comprehension task. In Association for Computational Linguistics (ACL), volume 1, pages 2358–2367.
19. Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading Wikipedia to answer open-domain questions. In Association for Computational Linguistics (ACL), volume 1, pages 1870–1879.
20. Karl Moritz Hermann, Toma´s Ko ˇ cisk ˇ y, Edward Grefenstette, Lasse Espeholt, Will Kay, ´ Mustafa Suleyman, and Phil Blunsom. 2015. Teaching machines to read and comprehend. In Advances in Neural Information Processing Systems (NIPS), pages 1693–1701.
21. Christopher Clark and Matt Gardner. 2018. Simple and effective multi-paragraph reading comprehension. In Association for Computational Linguistics (ACL), volume 1, pages 845–855.
22. Anthony Fader, Luke Zettlemoyer, and Oren Etzioni. 2014. Open question answering over curated and extracted knowledge bases. In SIGKDD Conference on Knowledge Discovery and Data Mining (KDD).
23. Lynette Hirschman, Marc Light, Eric Breck, and John D Burger. 1999. Deep read: A reading comprehension system. In Association for Computational Linguistics (ACL), pages 325–332.
24. Hsin-Yuan Huang, Chenguang Zhu, Yelong Shen, and Weizhu Chen. 2018b. FusionNet: Fusing via fully-aware attention with application to machine comprehension. In International Conference on Learning Representations (ICLR).
25. Robin Jia and Percy Liang. 2017. Adversarial examples for evaluating reading comprehension systems. In Empirical Methods in Natural Language Processing (EMNLP), pages 2021–2031
26. Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Association for Computational Linguistics (ACL), volume 1, pages 1601–1611.
27. Divyansh Kaushik and Zachary C. Lipton. 2018. How much reading does reading comprehension require? A critical investigation of popular benchmarks. In Empirical Methods in Natural Language Processing (EMNLP), pages 5010–5015.
28. Daniel Khashabi, Snigdha Chaturvedi, Michael Roth, Shyam Upadhyay, and Dan Roth. 2018. Looking beyond the surface: A challenge set for reading comprehension over multiple sentences. In North American Association for Computational Linguistics (NAACL), volume 1, pages 252–262.
29. Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. RACE: Large-scale reading comprehension dataset from examinations. In Empirical Methods in Natural Language Processing (EMNLP), pages 785–794.
30. Martin Raison, Pierre-Emmanuel Mazare, Rajarshi Das, and Antoine Bordes. 2018. ´Weaver: Deep co-encoding of questions and documents for machine reading. arXiv preprint arXiv:1804.10490.
31. Adams Wei Yu, Hongrae Lee, and Quoc Le. 2017. Learning to skim text. In Association for Computational Linguistics (ACL), volume 1, pages 1880–1890.
32. Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi. 2017. Bidirectional attention flow for machine comprehension. In International Conference on Learning Representations (ICLR).
33. Minjoon Seo, Sewon Min, Ali Farhadi, and Hannaneh Hajishirzi. 2018. Neural speed reading via Skim-RNN. In International Conference on Learning Representations (ICLR).
34. Matthew Richardson, Christopher J.C. Burges, and Erin Renshaw. 2013. MCTest: A challenge dataset for the open-domain machine comprehension of text. In Empirical Methods in Natural Language Processing (EMNLP), pages 193–203.
35. Hai Wang, Mohit Bansal, Kevin Gimpel, and David McAllester. 2015. Machine comprehension with syntax, frames, and semantics. In Association for Computational Linguistics (ACL), volume 2, pages 700–706.
36. Shuohang Wang and Jing Jiang. 2017. Machine comprehension using Match-LSTM and answer pointer. In International Conference on Learning Representations (ICLR).
37. Xiaodong Liu, Yelong Shen, Kevin Duh, and Jianfeng Gao. 2018. Stochastic answer networks for machine reading comprehension. In Association for Computational Linguistics (ACL), volume 1, pages 1694–1704.

##### Question Answering  

1. Eric Brill, Susan Dumais, and Michele Banko. 2002. An analysis of the AskMSR questionanswering system. In Empirical Methods in Natural Language Processing (EMNLP), pages 257–264.
2. Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. 2016. Learning to compose neural networks for question answering. In North American Association for Computational Linguistics (NAACL), pages 1545–1554.
3. Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra,C Lawrence Zitnick, and Devi Parikh. 2015. VQA: Visual Question Answering. In International Conference on Computer Vision (ICCV), pages 2425–2433.
4. Caiming Xiong, Victor Zhong, and Richard Socher. 2017. Dynamic coattention networks for question answering. In International Conference on Learning Representations
   (ICLR).
5. Shuohang Wang, Mo Yu, Jing Jiang, Wei Zhang, Xiaoxiao Guo, Shiyu Chang, Zhiguo
   Wang, Tim Klinger, Gerald Tesauro, and Murray Campbell. 2018b. Evidence aggregation for answer re-ranking in open-domain question answering. In International Conference on Learning Representations (ICLR).
   Wenhui Wang, Nan Yang, Furu Wei, Baobao Chang, and Ming Zhou. 2017. Gated selfmatching networks for reading comprehension and question answering. In Association
   for Computational Linguistics (ACL), volume 1, pages 189–198
6. Alon Talmor and Jonathan Berant. 2018. The web as a knowledge-base for answering complex questions. In North American Association for Computational Linguistics (NAACL),
   volume 1, pages 641–651.
7. Pum-Mo Ryu, Myung-Gil Jang, and Hyun-Ki Kim. 2014. Open domain question answering using Wikipedia-based knowledge model. Information Processing & Management,
   50:683–692.
8. Ellen Riloff and Michael Thelen. 2000. A rule-based question answering system for reading
   comprehension tests. In ANLP/NAACL Workshop on Reading comprehension tests as
   evaluation for computer-based language understanding sytems, pages 13–19.
9. Siva Reddy, Danqi Chen, and Christopher D Manning. 2019. CoQA: A conversational
   question answering challenge. Transactions of the Association of Computational Linguistics (TACL). Accepted pending revisions.
10. Yankai Lin, Haozhe Ji, Zhiyuan Liu, and Maosong Sun. 2018. Denoising distantly supervised open-domain question answering. In Association for Computational Linguistics (ACL), volume 1, pages 1736–1745.
11. Dan Moldovan, Sanda Harabagiu, Marius Pasca, Rada Mihalcea, Roxana Girju, Richard Goodrum, and Vasile Rus. 2000. The structure and performance of an open-domain question answering system. In Association for Computational Linguistics (ACL), pages 563–570.
12. Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. 2018. QuAC: Question answering in context. In Empirical Methods in Natural Language Processing (EMNLP), pages 2174–2184
13. Mohit Iyyer, Wen-tau Yih, and Ming-Wei Chang. 2017. Search-based neural structured learning for sequential question answering. In Association for Computational Linguistics (ACL), volume 1, pages 1821–1831.
14. Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know what you don’t know: Unanswerable questions for SQuAD. In Association for Computational Linguistics (ACL), volume 2, pages 784–789
15. Caiming Xiong, Victor Zhong, and Richard Socher. 2018. DCN+: Mixed objective and deep residual coattention for question answering. In International Conference on Learning Representations (ICLR).
16. Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Empirical Methods in Natural Language Processing(EMNLP), pages 2369–2380.
17. Xuchen Yao, Jonathan Berant, and Benjamin Van Durme. 2014. Freebase QA: Information extraction or semantic parsing? In ACL 2014 Workshop on Semantic Parsing, pages 82–86.
18. Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi, and Quoc V Le. 2018. QANet: Combining local convolution with global selfattention for reading comprehension. In International Conference on Learning Representations (ICLR).
19. Shuohang Wang, Mo Yu, Xiaoxiao Guo, Zhiguo Wang, Tim Klinger, Wei Zhang, Shiyu Chang, Gerald Tesauro, Bowen Zhou, and Jing Jiang. 2018a. Rˆ3: Reinforced readerranker for open-domain question answering. In Conference on Artificial Intelligence(AAAI).

##### Information Retrieval      



##### Dialogue Systems

1. Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. 2016. A diversitypromoting objective function for neural conversation models. In North American Association for Computational Linguistics (NAACL), pages 110–119.
2. Chia-Wei Liu, Ryan Lowe, Iulian Serban, Mike Noseworthy, Laurent Charlin, and Joelle Pineau. 2016. How NOT to evaluate your dialogue system: An empirical study of unsupervised evaluation metrics for dialogue response generation. In Empirical Methods in Natural Language Processing (EMNLP), pages 2122–2132.
3. Alexander Miller, Will Feng, Dhruv Batra, Antoine Bordes, Adam Fisch, Jiasen Lu, Devi Parikh, and Jason Weston. 2017. ParlAI: A dialog research software platform. In Empirical Methods in Natural Language Processing (EMNLP), pages 79–84.
4. Jianfeng Gao, Michel Galley, and Lihong Li. 2018. Neural approaches to conversational AI. arXiv preprint arXiv:1809.08267.
5. Daya Guo, Duyu Tang, Nan Duan, Ming Zhou, and Jian Yin. 2018. Dialog-to-action: Conversational question answering over a large-scale knowledge base. In Advances in Neural Information Processing Systems (NIPS), pages 2943–2952
6. Hsin-Yuan Huang, Eunsol Choi, and Wen-tau Yih. 2018a. FlowQA: Grasping flow in history for conversational machine comprehension. arXiv preprint arXiv:1810.06683.
7. Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason Weston. 2018. Personalizing dialogue agents: I have a dog, do you have pets too? In Association for Computational Linguistics (ACL), volume 1, pages 2204–2213

##### Natural language generation     

1. Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A human generated machine reading comprehension dataset. arXiv preprint arXiv:1611.09268.
2. Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Association for Computational Linguistics (ACL), volume 1, pages 889–898.

##### Summarization      

1. Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. Text
   Summarization Branches Out
2. Abigail See, Peter J Liu, and Christopher D Manning. 2017. Get to the point: Summarization with pointer-generator networks. In Association for Computational Linguistics (ACL), volume 1, pages 1073–1083.

##### Text classification      

1. Yoon Kim. 2014. Convolutional neural networks for sentence classification. In Empirical
   Methods in Natural Language Processing (EMNLP), pages 1746–1751.

##### Sentiment Analysis



##### Coreference resolution



##### Squence Labeling Problems:

1. [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](<https://arxiv.org/pdf/1603.01354.pdf>)

##### Named-entity recognition      



##### Part-of-speech tagging



##### Knowledge Graph

1. Amrita Saha, Vardaan Pahuja, Mitesh M. Khapra, Karthik Sankaranarayanan, and Sarath Chandar. 2018. Complex sequential question answering: Towards learning to converse over linked question answer pairs with a knowledge graph. In Conference on Artificial Intelligence (AAAI).



##### General NLP:

Regularization:

1. Yarin Gal and Zoubin Ghahramani. 2016. A theoretically grounded application of dropout in recurrent neural networks. In Advances in Neural Information Processing Systems (NIPS), pages 1019–1027.

Word/sentence representation:

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805), [BERT_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT_Torch.ipynb), 
2. Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. In North American Association for Computational Linguistics (NAACL), volume 1, pages 2227–2237.
   Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding by generative pre-training. Technical report, OpenAI.
3. Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pages 1532–1543.
4. Tomas Mikolov, Edouard Grave, Piotr Bojanowski, Christian Puhrsch, and Armand Joulin. 2017. Advances in pre-training distributed word representations. arXiv preprint arXiv:1712.09405

5. Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems (NIPS), pages 3111–3119.
6. Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. 2013. Semantic parsing on Freebase from question-answer pairs. In Empirical Methods in Natural Language Processing (EMNLP), pages 1533–1544.
7. Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. 2017. Enriching word vectors with subword information. Transactions of the Association of Computational Linguistics (TACL), 5:135–146
8. Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. 2017. Learned in translation: Contextualized word vectors. In Advances in Neural Information Processing Systems (NIPS), pages 6297–6308.
9. Jiatao Gu, Zhengdong Lu, Hang Li, and Victor O.K. Li. 2016. Incorporating copying mechanism in sequence-to-sequence learning. In Association for Computational Linguistics (ACL), pages 1631–1640.
10. Tao Lei, Yu Zhang, Sida I Wang, Hui Dai, and Yoav Artzi. 2018. Simple recurrent units for highly parallelizable recurrence. In Empirical Methods in Natural Language Processing (EMNLP), pages 4470–4481.
11. Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. 2016. Sequence level training with recurrent neural networks. In International Conference on Learning Representations (ICLR).

12. 
    Rupesh K Srivastava, Klaus Greff, and Jurgen Schmidhuber. 2015. Training very deep networks. In Advances in Neural Information Processing Systems (NIPS), pages 2377–2385.

13. Ilya Sutskever, Oriol Vinyals, and Quoc V Le. 2014. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems (NIPS), pages 3104–3112.
14. Christopher D Manning, Mihai Surdeanu, John Bauer, Jenny Finkel, Steven J Bethard, and David McClosky. 2014. The Stanford CoreNLP natural language processing toolkit. In Association for Computational Linguistics (ACL): System Demonstrations, pages 55–60.
15. Yoav Goldberg. 2017. Neural network methods for natural language processing, volume 10. Morgan & Claypool Publishers



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
5. <https://github.com/mhagiwara/100-nlp-papers>







