from flask import Flask, request, render_template
import os

app = Flask(__name__)

from commons import get_detection

@app.route('/', methods=['GET','POST'])
def predict():

	if request.method=='GET':
		return render_template('index.html')

	if request.method=='POST':
		if 'file' not in request.files:
			print('file not uploaded')
			return
		file = request.files['file']
		div_in = request.form['div_in']
		out = request.form['out']
		image = file.read()
		result,cant_total,cant_out = get_detection(image,div_in,out)

		return render_template('result.html',result = result,cant_total = cant_total,div_out = out, cant_out = cant_out )

if __name__ == '__main__':
	app.run(host='0.0.0.0')#host='0.0.0.0'
