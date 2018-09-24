import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import re
import gensim
import jieba
import tensorflow as tf
import numpy as np
import time
from random import shuffle
import os


class LstmModuleTrain():
    MAX_SIZE = 25
    dimsh = 0
    word2vec_model = None
    word2vec_path = 'D:/project/datainsights/NLP/data/baike/hanlp_model/baike_line_word_hanlp.model'
    # word2vec_path = '/home/gpu-s/cenzhongman/NLP/Word2vec/baike_line_word_hanlp.model'

    def __init__(self):
        # 加载vord2vec模型
        print('load vord2vec model ...')
        self.word2vec_model = gensim.models.word2vec.Word2Vec.load(self.word2vec_path)
        self.dimsh = self.word2vec_model.vector_size
        print('word2vrc model loaded finish, model dimensionality is', self.dimsh)

    def train(self, pos_path, neg_path, train_steps = 10001):
        num_nodes = 128
        batch_size = 16
        output_size = 2
        trainData, trainSteps, trainLabels = self.make_data(pos_path, neg_path)
        trainLabels = np.array(trainLabels)

        print("The trainData's shape is:", trainData.shape)
        print("The trainSteps's shape is:", trainSteps.shape)
        print("The trainLabels's shape is:", trainLabels.shape)

        # 2.实例化图对象
        graph = tf.Graph()
        with graph.as_default():
            tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, self.MAX_SIZE, self.dimsh))
            tf_train_steps = tf.placeholder(tf.int32, shape=(batch_size))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, output_size))

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

            train_logits = model(tf_train_dataset, tf_train_steps)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,
                                                        logits=train_logits))
            optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        summary_frequency = 1000

        # 3.实例化会话对象
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            mean_loss = 0

            # 保存模型max_to_keep是保存次数
            saver = tf.train.Saver(max_to_keep=5)

            # 若这个文件存在则加载模型
            if os.path.exists('ckpt/checkpoint'):
                print('加载模型文件')
                model_file = tf.train.latest_checkpoint('ckpt/')
                saver.restore(session, model_file)

            for step in range(train_steps):
                offset = (step * batch_size) % (len(trainLabels) - batch_size)
                feed_dict = {tf_train_dataset: trainData[offset:offset + batch_size],
                             tf_train_labels: trainLabels[offset:offset + batch_size],
                             tf_train_steps: trainSteps[offset:offset + batch_size]}
                _, l = session.run([optimizer, loss],
                                   feed_dict=feed_dict)
                mean_loss += l

                if step > 0 and step % summary_frequency == 0:
                    mean_loss = mean_loss / summary_frequency
                    print("The step is: %d" % (step))
                    print("In train data,the loss is:%.4f" % (mean_loss))
                    mean_loss = 0

                    # 保存模型
                    saver.save(session, 'ckpt/mnist.ckpt', global_step=step + 1)

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
    def get_words(self, path):
        line_lists = []
        file = open(path, encoding='utf-8', mode='r')
        lines = file.readlines()
        for line in lines:
            line = re.sub('[^\u4e00-\u9fa5a-zA-Z]', '', line)
            word_lists = jieba.lcut(line)
            line_lists.append(word_lists)
        # 返回一个文件中的所有评论的词组[['单词','单词2'],[],[]]
        file.close()
        return line_lists

    # 制作数据集，包括积极和消极数据，将会混合作为训练/测试的数据
    def make_data(self, pos_path, neg_path):
        pos_line_lists = self.get_words(pos_path)
        neg_line_lists = self.get_words(neg_path)
        pos_lines_array, pos_steps = self.words2array(pos_line_lists)  # 此处内存溢出
        neg_lines_array, neg_steps = self.words2array(neg_line_lists)
        datas, steps, labels = self.convert2Data(pos_lines_array, neg_lines_array, pos_steps, neg_steps)
        return datas, steps, labels
