import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from os import path
from tkinter.filedialog import Open

import json

import time

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-custom_6000',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from flask import Flask, render_template , request , jsonify

app = Flask(__name__)

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = './weights/transformerocr_25-07-2022.pth'
config['device'] = 'cpu'
config['cnn']['pretrained']=False
config['predictor']['beamsearch']=False
detector = Predictor(config)


# gui = Tk()
# gui.title('DEMO TKINTER')
# gui.geometry('1080x720')
# gui.attributes('-topmost',True)
# gui['bg'] = 'gray'
string_Path = ''
def getPath():

    if string_Path:
        print("Duong dan : "+string_Path)

    ftypes = [('Images file', '*.jpg'), ('All files', '*')]
    dlg = Open( filetypes = ftypes)
    print(dlg)
    fl = dlg.show()
    return fl
string_Path = getPath()
img_import = Image.open(string_Path)
resize = img_import.resize((416,416),Image.ANTIALIAS)
img = ImageTk.PhotoImage(resize)
print(img)
# sImage = Label(gui, text='', image=img)
# sImage.place(x= 4, y=5)
# btn_detectImage = Button(gui, text='CHOOSE FILE', width=15, height=2, bg='blue', command=getPath)
# btn_detectImage.place(x=250, y=450)

@app.route('/main' , methods=['POST'])
def index():
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    # btn_detectImage = Button(gui, text='Choose File', width=15, height=2, bg='blue', command=click)
    # btn_detectImage.place(x=250, y=450)

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    
    # gui = Tk()
    # gui.title('DEMO TKINTER')
    # gui.geometry('1080x720')
    # gui.attributes('-topmost',True)
    # gui['bg'] = 'gray'

    
    # filePath = filedialog.askopenfilename(initialdir='./')

    # btn_detectImage = Button(gui, text='DETECT', width=15, height=2, bg='blue', command=getPath)
    # btn_detectImage.place(x=250, y=450)
    
    # img_import = Image.open(stringPath)
    # resize = img_import.resize((416,416),Image.ANTIALIAS)
    # img = ImageTk.PhotoImage(resize)
    # print(img)
    # sImage = Label(gui, text='', image=img)
    # sImage.place(x= 4, y=5)


    input_size = 416 
    vid = cv2.imread(string_Path)
    frame_size = vid.shape[:2]
    image_data = cv2.resize(vid, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    start_time = time.time()


    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]
    
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # convert data to numpy arrays and slice out unused elements
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]


    original_h, original_w, _ = vid.shape
    bboxes = utils.format_boxes(bboxes, original_h, original_w)

    # store all predictions in one parameter for simplicity when calling functions
    pred_bbox = [bboxes, scores, classes, num_objects]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    allowed_classes = list(class_names.values())
    names = []
    
    # create list for final response
    response = []

    deleted_indx = []
    for i in range(num_objects):
        class_indx = int(classes[i])
        class_name = class_names[class_indx]
        if class_name not in allowed_classes:
            deleted_indx.append(i)
        else:
            names.append(class_name)
    names = np.array(names)
    count = len(names)

    bboxes = np.delete(bboxes, deleted_indx, axis=0)
    scores = np.delete(scores, deleted_indx, axis=0)



    list_cord = []
    for i in range(len(bboxes)):
        name = names[i]
        bbox = bboxes[i]
        list_cord.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
        # cv2.rectangle(vid, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (0, 255, 0), 2)
        # cv2.putText(vid, name, (int(bbox[0]), int(bbox[1])),0, 0.75, (255,0,0), 2)



    result = np.asarray(vid)
    string_bankNumber = ''
    string_bankName = ''
    string_userName = ''

    #result = cv2.resize(result,(800,800))
    for j, i in enumerate(list_cord):
        a, b, c, d = i
        img = result[b:b+d, a:a+c,:]
        # filename = './save/'+str(names[j])+'.jpg'
        # cv2.imwrite(filename, img)
        output = detector.predict(img)
        if names[j] == 'bank_number':
            string_bankNumber = output
        elif names[j] == 'bank_name':
            string_bankName = output
        elif names[j] == 'user_name':
            string_userName = output
        print(names[j]+": "+output)




    for i in range(len(bboxes)):
        name = names[i]
        bbox = bboxes[i]
        cv2.rectangle(vid, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (0, 255, 0), 2)
        cv2.putText(vid, name, (int(bbox[0]), int(bbox[1])),0, 0.75, (255,0,0), 2)
    result = cv2.resize(result,(416,416))
    filename = './save/Detect.jpg'
    cv2.imwrite(filename, result)
    # cv2.imshow("Output Image", result)
    # cv2.waitKey(0)
    
    label_BankName = Label(gui, text='Bank Name : ', font=('Fira Code',15))
    label_BankName.place(x= 200, y=500)
    sBankName = Label(gui, text=string_bankName, font=('Fira Code',15))
    sBankName.place(x= 350, y=500)

    label_BankNumber = Label(gui, text='Bank Number : ', font=('Fira Code',15))
    label_BankNumber.place(x= 200, y=530)
    sBankNumber = Label(gui, text=string_bankNumber, font=('Fira Code',15))
    sBankNumber.place(x= 370, y=530)

    label_Username = Label(gui, text='User Name : ', font=('Fira Code',15))
    label_Username.place(x= 200, y=560)
    sUserName = Label(gui, text=string_userName, font=('Fira Code',15))
    sUserName.place(x= 350, y=560)

    imgMuiTen = Image.open('muitenfg.png')
    resize = imgMuiTen.resize((180,70),Image.LANCZOS)
    imgMuiTen = ImageTk.PhotoImage(resize)
    print(imgMuiTen)
    sImageMuiTen = Label(gui, text='', image=imgMuiTen)
    sImageMuiTen.place(x= 430, y=150)


    imgDetect = Image.open(filename)
    resize = imgDetect.resize((416,416),Image.LANCZOS)
    imgDetect = ImageTk.PhotoImage(resize)
    print(imgDetect)
    sImageDetect = Label(gui, text='', image=imgDetect)
    sImageDetect.place(x= 640, y=5)

    # print(json.dumps({"bank_name":string_bankName,"bank_number":string_bankNumber,"user_name":string_userName}))

    return jsonify({"bank_name":string_bankName,"bank_number":string_bankNumber,"user_name":string_userName})