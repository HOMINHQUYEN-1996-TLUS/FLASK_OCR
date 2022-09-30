import time
from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64

from sqlalchemy import false
from yolo_detection_images import runModel,runOCR
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)

dem = 0
############################################## THE REAL DEAL ###############################################
@app.route('/detectObject' , methods=['POST'])
def mask_image():

	while(True):
		# print(request.files , file=sys.stderr)
		file = request.files['image'].read() ## byte file
		start_time = time.time()
		npimg = np.fromstring(file, np.uint8)
		img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
		string_bankNumber,string_bankName,string_userName = runOCR(img)
		######### Do preprocessing here ################
		# img[img > 150] = 0
		## any random stuff do here
		################################################

		
		
		img = Image.fromarray(img.astype("uint8"))
		rawBytes = io.BytesIO()
		img.save(rawBytes, "JPEG")
		rawBytes.seek(0)
		img_base64 = base64.b64encode(rawBytes.read())
		global dem
		dem += 1
		end_time = time.time()
		elapsed_time = end_time - start_time
		# return jsonify({'status':str(img_base64)},{'string_bankNumber':string_bankNumber},
		# 			{'string_bankName':string_bankName},{'string_userName':string_userName})
		Data = {'data':{'id':str(dem)+'_'+str(start_time)[6:12],'img_base64':str(img_base64),'string_bankNumber':string_bankNumber,
					'string_bankName':string_bankName,'string_userName':string_userName,'time':format(elapsed_time)[0:4]}}
		return jsonify(Data)
##################################################### THE REAL DEAL HAPPENS ABOVE #####################################
@app.route('/test' , methods=['GET','POST'])
def test():
	print("log: got at test" , file=sys.stderr)
	return jsonify({'status':'succces'})

@app.route('/')
def home():
	return render_template('./index.html')

	
@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
	app.run(host='0.0.0.0',debug=false)
	#app.run(ssl_context=('cert.pem', 'key.pem'))
	#app.run(ssl_context='adhoc')
