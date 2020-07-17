import re
import jieba
import sys
import matplotlib
import jieba.posseg as ps
from nltk import *
from matplotlib import rcParams
from matplotlib.font_manager import findfont, FontProperties, _rebuild
from universalMethod import *

# 读取文本信息
def readFile(path):
    str_doc = ""
    with open(path, 'r', encoding='utf-8') as f:
        str_doc = f.read()
    return str_doc

# 正则对字符串清洗
def textParse(str_doc):
    # 去掉字符
    str_doc = re.sub('\u3000', '', str_doc)
    # 去除空格
    str_doc = re.sub('\s+', ' ', str_doc)
    # 去除换行符
    str_doc= re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', " ", str_doc)
    # 正则过滤掉特殊符号,标点,英文,数字...
    r1 = '[a-zA-Z0-9]+'
    str_doc = re.sub(r1, "", str_doc)
    return str_doc

def get_stop_words(path=r'./停用词.txt'):
    file = open(path, 'r', encoding='utf-8').read().split('\n')
    return set(file)

def rm_tokens(words, stwlist):
    words_list = list(words)
    stop_words = stwlist
    for i in range(words_list.__len__())[::-1]:
        # 去除停用词
        if words_list[i] in stop_words:
            words_list.pop(i)
        # 去除数字
        elif words_list[i].isdigit():
            words_list.pop(i)
        # 去除单个字符
        elif len(words_list[i]) == 1:
            words_list.pop(i)
        # 去除空格
        elif words_list[i] == " ":
            words_list.pop(i)
    return words_list

def seg_doc(str_doc):
    # 1.处理原文本
    sent_list = str_doc.split('\n')
    sent_list = map(textParse, sent_list)
    # 2.获取停用词
    stwlist = get_stop_words()
    # 3.分词并去除停用词
    word_2dlist = [rm_tokens(jieba.cut(part, cut_all=False), stwlist) for part in sent_list]
    # 4. 合并列表
    word_list = sum(word_2dlist, [])
#     print(sent_list)
    return word_list

def nltk_wf_feature(word_list=None):
    # 方法一: 得到的关键词和词频不是一一对应的
    fdist = FreqDist(word_list)
#     print(fdist.keys(), "\n", fdist.values(), "\n")
    
    # 查看指定词语词频
    w = "陈奕迅"
    print(w, "Frequency: ", fdist.freq(w))
    print(w, "Num of app: ", fdist[w])
    
    # 频率分布表
    print('='*3, "频率分布表", '='*3)
    fdist.tabulate(10)
    
    print('='*3, "频率分布图", '='*3)
    fdist.plot(30)
    
    return fdist
    
    # 解决中文显示问题
    # 1.查看当前字体
    # 2.更换字体库

def nltk_wf_feature2(word_list=None):
    # 方法二:
    from collections import Counter
    words = Counter(word_list)
    print(words.keys(), "\n", words.values())
    print("------根据字符长度------")
    wlist = [w for w in words if len(w) > 2]
    print(wlist)
    
def hl_freqWord(fdist):
    wordList = []
    print('='*3, '打印统计的词频', '='*3)
    for key in fdist.keys():
        if fdist.get(key) > 2 and fdist.get(key) < 20:
            wordList.append(key + ":" + str(fdist.get(key)))
    return wordList

def extract_featureWord(str_doc):
    featureWords = ""
    stwList = get_stop_words()
    # user defined the property lis of feature word: 人名, 地名, 机构名
    user_pos_list = ['nr', 'ns', 'nt', 'nz']
    for word, pos in ps.cut(str_doc):
        if word not in stwList and pos in user_pos_list:
            if word + ' ' + pos + '\n' not in featureWords:
                featureWords += word + ' ' + pos + '\n'
    print('\n命名实体识别:\n')
    print(featureWords)
    return featureWords

def sklearn_tfidf_feature(corpus=None):
    # 构建词汇表
    vectorizer = CountVectorizer()
    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋模型中的所有词语
    words = vectorizer.get_feature_names()
    # 将tf-idf矩阵抽取出来, 元素a[i][j]表示j词在i类文本中的权重
    weight = tfidf.toarray()
    for i in range(len(weight)):
        print(u"----这里输出第", i, u"类文本的词语tf-idf权重")
        for j in range(len(words)):
            print(words[j], weight[i][j])
