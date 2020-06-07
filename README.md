# Word2VecSeq2SeqBaseLineAll
Word2VecSeq2SeqBaseLineAll

（1）此项目基于百度竞赛“问答摘要与推理”，竞赛简介即数据获取地址：https://aistudio.baidu.com/aistudio/competition/detail/3  
（2）dataProcess为数据处理，从原始数据开始切词、分词、去除停用词、去除噪声词。最后得到词表、词向量矩阵等。  
（3）工程目录下seq2seq_tf2_d、seq2seq_pgn_tf2_d、seq2seq_tf2_z、seq2seq_pgn_tf2_z四个模块是相互独立的。  
（4）seq2seq_tf2_d只实现了seq2seq常规方法，seq2seq_pgn_tf2_d实现了PGN方法。  
（5）在跑每个模块的代码时先将dataProcess下dataSetCommon中的训练数据、测试数据、vocab复制到模块下的dataset中。
