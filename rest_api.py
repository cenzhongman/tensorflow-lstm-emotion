from flask import Flask
from flask import make_response
from flask import request

import test_lstm

app = Flask(__name__)

tester = test_lstm.LstmTest()


@app.route('/')
def hello_world():
    return 'Hello World!'


def degree_format(pos, neg):
    if pos > neg:
        return pos
    else:
        return -neg


@app.route('/emotion_result/', methods=['GET', 'POST', 'OPTIONS'])
def emotion():
    if request.method == 'POST' :
        data = request.get_json()
        text = data['text']
        predictions = tester.test([text])
        pos_degree = predictions[0][0]
        neg_degree = predictions[0][1]
        degree = degree_format(float(pos_degree), float(neg_degree))
        rst = make_response(str(degree) + '|' + text)
        rst.headers['Access-Control-Allow-Origin'] = '*'
        rst.headers['Access-Control-Allow-Headers'] = 'X-Requested-With,accept, origin, content-type'
        return rst
    elif request.method == 'OPTIONS':
        rst = make_response('')
        rst.headers['Access-Control-Allow-Origin'] = '*'
        rst.headers['Access-Control-Allow-Headers'] = 'X-Requested-With,accept, origin, content-type'
        return rst
    else:
        return 'You must post some data!'


# 旧的运行方式
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8004)
