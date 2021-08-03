import gensim
from gensim.models import Word2Vec
import jieba
import jieba.posseg as pseg
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

# 加载词向量模型
model= KeyedVectors.load_word2vec_format(datapath('D:\Downloads\sgns.renmin.bigram-char.bz2'), binary=False)

# 计算词表中的同序异素词的相似度
for line in open('./wordlist.txt', 'r', encoding='utf-8'):
    line = line.strip('\n')
    words = pseg.cut(line)
    for word, flag in words:
        print(word, flag)
    first_word, second_word = line.split('/', 2)
    if first_word not in model.index2word or second_word not in model.index2word:
        #print(first_word)
        continue
    else:
        #print(first_word, second_word)
        y = model.similarity(first_word, second_word)
        print(first_word, second_word, y)