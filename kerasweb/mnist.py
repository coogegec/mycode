from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app=Flask(__name__)

@app.route('/')
def main():
    return render_template('mnist/index.html')

@app.route('/uploader', methods=['POST'])
def upload_image():
    model=load_model('d:/data/mnist')
    img=Image.open(request.files['file'].stream).convert('L') # 업로드한 파일을 gray scale로 변환
    print(type(img))
    img=img.resize((28,28)) # 업로드한 파일 사이즈가 mnist 이미지 사이즈와 같도록 처리
    arr=np.array(img)/255 # 넘파이 배열로 변환
    print(arr.shape)
    arr=arr.reshape(1,28,28,1) # keras 모형에서 읽을 수 있도록 28×28에서 1×28×28×1로 차원 변경
    # ※ 이미지 개수 × 가로 사이즈 × 세로 사이즈 × 흑백(1)/컬러(3)
    pred=model.predict(arr)
    pred=np.argmax(pred, axis=1)
    return '숫자 이미지 : '+str(pred[0])

if __name__=='__main__':
    app.run(port=8000, threaded=False)