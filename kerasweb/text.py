from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import gensim
from konlpy.tag import Okt

app=Flask(__name__)

MIN_NOUN_VECTOR_VALUE=-10.0
MAX_NOUN_VECTOR_VALUE=10.0
NOUN_VECTOR_SIZE=100
TITLE_LENGTH=37
# 텍스트의 길이가 짧을 경우 길이를 맞추기 위해 패딩 처리
def generate_random_noun_vector():
    return np.random.uniform(low=MIN_NOUN_VECTOR_VALUE, high=MAX_NOUN_VECTOR_VALUE,
                             size=(NOUN_VECTOR_SIZE,))

def generate_zero_noun_vector():
    return np.random.uniform(low=0.0, high=0.0, size=(NOUN_VECTOR_SIZE,))

@app.route('/')
def home():
    return render_template('text/index.html')

@app.route('/text_result', methods=['POST'])
def result():
    model=load_model('d:/data/text')
    text=request.form['text']

    twitter_nlp=Okt()
    title_noun_arr=[]
    title_noun_arr.append(twitter_nlp.nouns(text)) # 명사만 추출

    # 벡터 모델을 불러와서 텍스트를 벡터로 변환
    w2v_model=gensim.models.Word2Vec.load('d:/data/text/text_100.model')
    title_noun_vector_arr=[]
    for index, title_nouns in enumerate(title_noun_arr):
        noun_vector_arr=[]
        for noun in title_nouns:
            try:
                noun_vector=w2v_model[noun] # 단어별로 벡터 모델 적용
            except:
                noun_vector=generate_random_noun_vector()
            noun_vector_arr.append(noun_vector)
        title_noun_vector_arr.append(noun_vector_arr)
    # 인덱스와 요소를 하나씩 전개
    for index, title_noun_vector in enumerate(title_noun_vector_arr):
        while len(title_noun_vector)<TITLE_LENGTH:
            title_noun_vector.append(generate_zero_noun_vector())
        title_noun_vector=title_noun_vector[:TITLE_LENGTH]
    noun_vector_arr=np.array(title_noun_vector_arr).reshape(1,37,100,1)
    pred=model.predict(noun_vector_arr)
    a=pred[0][0]
    result=''
    if a>=0.5:
        result='긍정'
    else:
        result='부정'
    return render_template('text/result.html', result=result, rate=f'{a*100:.2f}%', review=text)

if __name__=='__main__':
    app.run(port=8000, threaded=False)