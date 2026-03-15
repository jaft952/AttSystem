import os,cv2,glob
RAW='data/raw'
FINAL='data/pipeline2/final_processed'
DNN_PROTO='models/face_detector/deploy.prototxt'
DNN_MODEL='models/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'

def list_dirs(root):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])

raw_people=list_dirs(RAW) if os.path.exists(RAW) else []
final_people=list_dirs(FINAL) if os.path.exists(FINAL) else []
print(f'raw_people={len(raw_people)}, final_people={len(final_people)}')
print('People missing in final:', [p for p in raw_people if p not in final_people])

has_dnn=os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL)
net=None
if has_dnn:
    try:
        net=cv2.dnn.readNet(DNN_MODEL, DNN_PROTO)
    except Exception as e:
        print('Failed to load DNN:', e)

closeups=[]
THRESH=0.5
for person in final_people:
    imgs=glob.glob(os.path.join(FINAL,person,'*_face_processed.jpg'))
    for p in imgs:
        img=cv2.imread(p)
        if img is None: continue
        h,w=img.shape[:2]
        # detect face
        if net is None:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
            faces=face_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=3)
            if len(faces)==0:
                continue
            x,y,fw,fh=sorted(faces,key=lambda r:r[2]*r[3],reverse=True)[0]
        else:
            blob=cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
            net.setInput(blob)
            det=net.forward()
            best_conf=0; best_box=None
            for i in range(det.shape[2]):
                conf=float(det[0,0,i,2])
                if conf>best_conf:
                    box=det[0,0,i,3:7]*[w,h,w,h]
                    best_conf=conf; best_box=box
            if best_box is None or best_conf<0.4:
                continue
            l,t,r_,b=best_box.astype(int)
            x=max(0,l); y=max(0,t); fw=max(0,min(w,r_)-x); fh=max(0,min(h,b)-y)
        area=(fw*fh)/(w*h) if w*h>0 else 0
        if area>THRESH:
            closeups.append((person,os.path.basename(p),round(area,3)))

print('\nClose-up detections (area_ratio>0.5):')
for c in closeups:
    print(c)
print('Total close-ups:', len(closeups))
