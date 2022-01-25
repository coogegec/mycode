from flask import Flask, Markup, render_template, request
app=Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    return render_template('gugu.html')

@app.route('/gugu_result', methods=['POST'])
def gugu_result():
    dan=int(request.form['dan']) # form 변수 (name="dan")
    result=''
    for i in range(1,10):
        result += f'{dan} × {i} = {dan*i}<br>'
    result=Markup(result) # Markup() : html 태그를 인식시킴
    return render_template('gugu_result.html', result=result)

if __name__ == '__main__':
    # threaded를 True로 설정하면 신경망 모형을 불러오는 코드에서 에러가 발생하므로 False로 설정
    app.run(port=80, threaded=False)