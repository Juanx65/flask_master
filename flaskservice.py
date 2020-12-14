from flask import Flask, request, render_template
import os

app = Flask(__name__)

from commons import get_detection

@app.route('/', methods=['GET','POST'])
def predict():

	if request.method=='GET':
		return render_template('index.html', value="!!!!!")

	if request.method=='POST':
		predict_bill = "1kbill"
		if 'file' not in request.files:
			print('file not uploaded')
			return
		file = request.files['file']
		image = file.read()
		result,cant_total = get_detection(img=image)
		return render_template('result.html',result = result,cant_total = cant_total)

if __name__ == '__main__':
	app.run(debug=True)
