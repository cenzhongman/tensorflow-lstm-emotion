import os

import test_lstm
import train_lstm_model

# 训练

# 加载可以训练的文件
# 获取文件，此处的训练文件应该是一一对应的
# root_path = 'D:/project/datainsights/NLP/data/JDComments/'
# # root_path = '/home/gpu-s/cenzhongman/NLP/JDComments/'
# pos_train_file_list = os.listdir(root_path + 'train/pos/')
# neg_train_file_list = os.listdir(root_path + 'train/neg/')
# trainer = train_lstm_model.LstmModuleTrain()
# for i in range(len(pos_train_file_list)):
#     # 训练一个文件
#     print('训练一个文件')
#     trainer.train(root_path + 'train/pos/' + pos_train_file_list.pop(),
#                   root_path + 'train/neg/' + neg_train_file_list.pop())

# 测试
tester = test_lstm.LstmTest()
predictions = tester.test(['电脑真的很差， 拿到手屏幕就是坏的，明显是生产不过关，还要我出检验报告。感觉你们京东知道是坏的还故意发出来。退换货还那么麻烦。。'])
for prediction in predictions:
    pos_degree = prediction[0]
    neg_degree = prediction[1]
    print(pos_degree, neg_degree)
