import urllib.request

url_model = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
urllib.request.urlretrieve(url_model, r"C:\Users\jaft9\School\AttSystem\models\ssd\res10.caffemodel")

url_proto = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
urllib.request.urlretrieve(url_proto, r"C:\Users\jaft9\School\AttSystem\models\ssd\deploy.prototxt")