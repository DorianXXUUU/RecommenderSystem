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
            
            
            
            
 # 构建语料词典 -- 通过corpora
def gensim_Corpus(corpus=None):
    # 1. 词典
    dictionary = corpora.Dictionary(corpus)
    print(dictionary)
    # 2. 仅出现一次的词
    once_ids = [tokenId for tokenId, docFreq in dictionary.dfs.items() if docFreq==1]
    # print('仅出现一次的词: \n', once_ids)
    # 过滤掉只出现过一次的词
    dictionary.filter_tokens(once_ids)
    print('自定义选择的特征词典: \n', dictionary)
    # 为unique tokens重新分配id
    dictionary.compactify()
    
    # 3. 存储词典
    savePath = r'/Users/xudeyan/Desktop/NLP/dataSet/imooc/mycorpus.dict'
    dictionary.save(savePath)
    
    # 加载字典
    # mydict = corpora.Dictionary.load(savePath)
    # print("加载DICT词典: \n", mydict)
    
    # 4. TXT 方法储存
    savePath = r'/Users/xudeyan/Desktop/NLP/dataSet/imooc/mycorpus.txt'
    dictionary.save_as_text(savePath)
    
    # list2 = corpora.Dictionary.load_from_text(savePath)
    # print(list2)

# 创建数据集
def loadDataSet():
    corpus = []
    tiyu = ['曼联', '大将', '终解', '牵挂', '白色', '鲜花', '寄托', '沉痛', '哀悼', '本报记者',
            '预赛', '荷兰队', '首发', '强阵', '利物浦', '双煞', '撑起', '三叉戟', '新浪', '体育讯',
            '卡帅', '防守', '死里逃生', '德克', '配得', '伟大', '新浪', '体育讯', '北京', '时间',
            '韩佳良', '韩国', '选手', '是否', '故意', '很难说', '感谢', '支持', '本报讯', '记者',
            '图文', '天津', '新疆', '艾伦', '单手', '上篮', '体育', '篮球', '新疆', '广汇胜',
            '罗斯', '开局', '不顺', '责任', '控制', '失误', '热火', '新浪', '体育讯', '北京',
            '斯巴鲁', '车队', '小将', '露头', '裴亮', '孙超', '指引', '赛车', '新浪', '体育讯',
            '北京', '球迷', '赛后', '口水', '京骂', '伺候', '威尔斯', '棒子', '杂物', '袭击',
            '湖人', '怪才', '篮板', '不可思议', '命中', '睡得', '这球', '惊醒', '湖人队', '球迷',
            '朱芳雨', '腰伤', '前往', '香港', '复查', '宏远', '北上', '浙江', '缺阵', '本报',
            '法拉利', '主席', '蒙总', '放出', '豪言', '必须', '拿下', '今年', '冠军', '新浪',
            '滨岩答', '网友', '皇马', '崩溃', '其实', '正常', '不败', '水分', '很大', '西甲',
            '斯诺克', '美女', '裁判', '诠释', '迷人', '真意', '性感', '更爱', '专业', '记者',
            '热门', '占据', '训练场', '当地人', '不知', '西班牙', '驾临', '记者', '李镇伯', '发自',
            '上海', '奥沙利', '爆冷', '无缘', '汉密尔顿', '破百', '晋级', '新浪', '体育讯', '北京',
            '金羽', '取代', '祁宏', '辅导', '女足', '李霄鹏力', '邀大羽', '指导', '进攻', '本报讯',
            '列日', '阿森纳', '首发', '爱德华多', '先发', '中场', '大将', '复出', '新浪', '体育讯',
            '高洪波', '感谢', '队员', '挺过', '难关', '鼓励', '勇敢', '面对', '世界杯', '新浪',
            '详解', '球砸勒夫', '原因', '过节', '风格', '新浪', '体育讯', '昨天', '战胜', '森林狼',
            '韩乔生', '炮轰', '臭不可闻', '这帮', '人全', '赶回', '家去', '前天', '最具', '娱乐',
            '魔兽', '黑手', '韦德', '空中', '米平', '顽强', '詹姆斯', '次节', '开局', '先行'
           ]
    
    yule = ['韩寒', '香港', '书展', '关注', '幽默', '回击', '陈文茜', '批评', '亚洲', '规模'
          ,'梁洛施', '产后', '公开', '亮相', '依然', '青春', '日前', '香港', '举行', '郑秀文'
          ,'陈慧琳', '享受', '寂寞', '婚后', '个人空间', '陈慧琳', '听歌', '幸福', '出嫁', '刘太'
          ,'余文乐', '间接', '承认', '分手', '坏脾气', '险酿', '冲突', '组图', '新浪', '娱乐'
          ,'组图', '林峰', '帅气', '代言', '珠宝', '拒绝', '穿耳洞', '影响', '运气', '新浪'
          ,'刘德华', '首谈', '新婚', '生活', '本书', '来教', '大众', '刚刚', '过去', '刘德华'
          ,'胡歌', '潮人', '打扮', '亮相', '机场', '低调', '吃快餐', '组图', '新浪', '娱乐'
          ,'化身', '排球', '宝贝', '沙滩', '裸晒', '新浪', '娱乐', '海宁', '两件', '泳衣'
          ,'于震', '黑白', '现身', '颁奖礼', '压轴', '出场', '追捧', '新浪', '娱乐', '第三届'
          ,'黄秋生', '忙于', '照顾', '儿子', '谢婷婷', '不知', '老爸', '失恋', '新浪', '娱乐'
          ,'组图', '绯闻', '女孩', '性感', '大变身', '上秀', '乳沟', '秀美', '新浪', '娱乐'
          ,'甘薇', '美俏', '分享', '血拼', '心得', '亲授', '打折', '购物', '宝典', '新浪'
          ,'傅颖', '香港', '书展', '连夜', '签书超', '二千', '傅颖', '昨日', '书展', '宣传'
          ,'陈慧琳', '今日', '出嫁', '父亲', '否认', '准备', '千万', '嫁妆', '本报讯', '记者'
           ]
    jiaoyu = ['全国', '万人', '报名', '高考', '平均', '录取率', '提高', '新华社', '北京', '日电',
              '语种', '优势', '凸显', '欧洲', '成为', '留学', '热门', '目的地', '日前', '记者',
              '公务员', '报考', '冷热', '不均', '职位', '无人问津', '国考', '网上', '报名', '进入',
              '留学生', '讲述', '韩国', '饭店', '打工', '心酸', '经历', '韩国', '首尔', '韩国',
              '江苏', '初中', '试水', '双分制', '尺子', '学生', '本报讯', '记者', '苏婷', '常规',
              '清华', '浙大', '六校', '明年', '联考', '试题', '整体', '难度', '将会', '下降',
              '重庆', '中职', '学校', '招生', '持证', '上岗', '本报讯', '记者', '秦健', '我市',
              '图文', '世通', '北京', '恭贺', '移民', '频道', '上线', '贺词', '新浪', '教育',
              '北京', '中招', '首批', '录取', '所校', '提前', '录取', '京报网', '今天', '记者',
              '新生', '报考', '自考', '前先', '备齐', '相关', '自考', '教材', '北京', '考试',
              '韩国', '英语', '培训', '市场', '正处', '黄金岁月', '近几年', '韩国', '几家', '经营',
              '成都市', '部分', '省级重点', '中学', '高中', '招生', '缺额', '记者', '昨天', '成都市',
              '福建师范大学', '高考', '录取', '查询', '系统', '开通', '新浪', '教育', '福建师范大学', '高考',
              '高招', '常规', '录取', '结束', '湖北省', '考生', '大学', '湖南省', '高招', '常规',
              '北京舞蹈学院', '招生', '章程', '新浪', '教育', '北京舞蹈学院', '招生', '章程', '公布', '以下',
              '河南', '中医学院', '计划', '招生', '记者', '李红', '该校', '河南', '招生', '代码'
              ]

    shizheng =  ['媒体', '列举', '意大利', '现任', '总理', '屹立', '不倒', '五大', '理由', '丑闻'
          ,'意大利', '游轮', '保安', '开枪', '击退', '索马里', '海盗', '中新网', '日电', '新加坡'
          ,'泰国', '曼谷', '步兵团', '炸弹', '袭击', '士兵', '受伤', '新华网', '快讯', '设在'
          ,'奥地利', '总统', '菲舍尔', '再次', '赢得', '总统', '选举', '中国日报', '消息', '奥地利'
          ,'阿尔巴尼亚', '发生', '里氏', '地震', '新华网', '地拉那', '日电', '记者', '杨柯', '阿尔巴尼亚'
          ,'美政府', '清理', '漏油', '说法', '夸大', '事实', '国际', '在线', '专稿', '卫报'
          ,'数字', '盘点', '威廉', '王子', '婚礼', '当日', '敲响', '钟声'
        ]
    
    keji =  ['完美', '时尚', '卡片机', '索尼', '一下', '跌破', '泡泡网', '上海', '分站', '索尼'
     ,'支持', '光学', '变焦', '富士', '仅售', '作者', '玛琪雅朵', '富士', '搭载', '一枚'
     ,'时尚', '炫彩', '音效', '容量', '三星', '作者', '上海', '行情', '三星', '三星'
     ,'做客', '都市', '声数码', '消费', '栏目', '相机', '问答', '集锦', '新浪', '数码'
     ,'高性能', '入门', '单反', '索尼', '防抖', '套机', '送卡', '编辑', '观点', '具备'
     ,'价格', '小降', '富士', '变焦', '促销', '泡泡网', '数码相机', '频道', '富士', '搭载'
     ,'奥林巴斯', '产品', '发布会', '抢先', '亮相', '泡泡网', '数码相机', '频道', '奥林巴斯', '即将'
     ,'十万', '网商', '在线', '公开', '企业', '身份', '信息', '晨报讯', '中国', '商品'
     ,'行情', '主流', '液晶电视', '直降', '系列', '全新', '产品', '阵营', '最为', '热销'
     ,'高性能', '商用', '投影', '爱普生', '爱普生', '身材', '更加', '小巧', '轻盈', '丝毫'
     ,'价格', '绝对', '坚挺', '佳能', '二代', '镜头', '小降', '北京', '行情', '佳能'
        ]
    
    corpus.append(tiyu)
    corpus.append(yule)
    corpus.append(jiaoyu)
    corpus.append(shizheng)
    corpus.append(keji)
    
    classVec = ['体育', '娱乐', '教育', '时政', '科技']
    
    return corpus, classVec

