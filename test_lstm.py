import warnings
import gensim
import jieba
import re
import os
import tensorflow as tf
import numpy as np
from random import shuffle

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


class LstmTest():
    MAX_SIZE = 25
    dimsh = 0
    word2vec_model = None
    # word2vec_path = 'D:/project/datainsights/NLP/data/baike/hanlp_model/baike_line_word_hanlp.model'
    word2vec_path = '/home/gpu-s/cenzhongman/NLP/Word2vec/baike_line_word_hanlp.model'
    # word2vec_path = '/home/ubuntu/project/data/model/hanlp_model/baike_line_word_hanlp.model'

    def __init__(self):
        # 加载vord2vec模型
        print('load vord2vec model ...')
        self.word2vec_model = gensim.models.word2vec.Word2Vec.load(self.word2vec_path)
        self.dimsh = self.word2vec_model.vector_size
        print('word2vrc model loaded finish, model dimensionality is', self.dimsh)

    def test(self, texts):
        testData, testSteps = self.make_data(texts)

        num_nodes = 128

        graph = tf.Graph()
        with graph.as_default():
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_nodes,
                                                     state_is_tuple=True)

            w1 = tf.Variable(tf.truncated_normal([num_nodes, num_nodes // 2], stddev=0.1))
            b1 = tf.Variable(tf.truncated_normal([num_nodes // 2], stddev=0.1))

            w2 = tf.Variable(tf.truncated_normal([num_nodes // 2, 2], stddev=0.1))
            b2 = tf.Variable(tf.truncated_normal([2], stddev=0.1))

            def model(dataset, steps):
                outputs, last_states = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                         dtype=tf.float32,
                                                         sequence_length=steps,
                                                         inputs=dataset)
                hidden = last_states[-1]

                hidden = tf.matmul(hidden, w1) + b1
                logits = tf.matmul(hidden, w2) + b2
                return logits

            tf_test_dataset = tf.constant(testData, tf.float32)
            tf_test_steps = tf.constant(testSteps, tf.int32)

            test_prediction = tf.nn.softmax(model(tf_test_dataset, tf_test_steps))

        with tf.Session(graph=graph) as session:
            # 保存模型max_to_keep是保存次数
            saver = tf.train.Saver(max_to_keep=5)

            # 若这个文件存在则加载模型
            if os.path.exists('ckpt/checkpoint'):
                print('加载模型文件')
                model_file = tf.train.latest_checkpoint('ckpt/')
                saver.restore(session, model_file)
            predictions = session.run(test_prediction)

            return predictions

            # acrc = 0
            # for i in range(len(prediction)):
            #     if prediction[i][testLabels[i].index(1)] > 0.5:
            #         acrc = acrc + 1
            #         print(prediction[i][testLabels[i].index(1)])
            # print("In test data,the accuracy is:%.2f%%" % ((acrc / len(testLabels)) * 100))

    # 将[words,words]转化为linesArray, steps
    def words2array(self, line_lists):
        lines_array = []
        steps = []
        for word_lists in line_lists:
            words_array = []
            # 拥有词向量的统计
            p = 0
            # 没有词向量的统计
            t = 0
            # 只使用前MAX_SIZE个词语
            for i in range(self.MAX_SIZE):
                if i < len(word_lists):
                    try:
                        words_array.append(self.word2vec_model.wv.word_vec(word_lists[i]))
                        p = p + 1
                    except KeyError:
                        t = t + 1
                        continue
                else:
                    # 词组本来就小于MAX_SIZE个，用0.0向量填充
                    words_array.append(np.array([0.0] * self.dimsh))
            # 弥补词向量中没有的单词，用0.0向量填充
            for i in range(t):
                words_array.append(np.array([0.0] * self.dimsh))
            # 有效长度
            steps.append(p)
            lines_array.append(words_array)
        # 将lines_array转为真正的矩阵{[(1,4,4),(),()],[],[]}
        lines_array = np.array(lines_array)
        # 每句效的词数steps[4,7,8,13]
        steps = np.array(steps)
        return lines_array, steps

    # 混淆积极和消极数据，思路，先组合，调整顺序，取消组合
    def convert2Data(self, posArray, negArray, posStep, negStep):
        # 用来装所有东西的容器
        item = []
        data = []
        steps = []
        labels = []
        for i in range(len(posArray)):
            item.append([posArray[i], posStep[i], [1, 0]])
        for i in range(len(negArray)):
            item.append([negArray[i], negStep[i], [0, 1]])
        shuffle(item)
        for i in range(len(item)):
            data.append(item[i][0])
            steps.append(item[i][1])
            labels.append(item[i][2])
        # {[(1,4,4),(),()],[],[]}
        data = np.array(data)
        # steps(4,7,8,13)
        steps = np.array(steps)
        # labels[(0,1),(1,0)]
        return data, steps, labels

    # 将文件变成词组，输入路径，返回[words,words]
    def get_words(self, texts):
        line_lists = []
        for text in texts:
            text = re.sub('[^\u4e00-\u9fa5a-zA-Z]', '', text)
            word_lists = jieba.lcut(text)
            line_lists.append(word_lists)
        # 返回一个文件中的所有评论的词组[['单词','单词2'],[],[]]
        return line_lists

    # 制作数据集，包括积极和消极数据，将会混合作为训练/测试的数据
    def make_data(self, texts):
        line_lists = self.get_words(texts)
        lines_array, steps = self.words2array(line_lists)
        return lines_array, steps
