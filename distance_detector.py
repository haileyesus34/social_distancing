from ultralytics import YOLO
import cv2
import math 
from utils import measure_distance
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 980)

# model
model = YOLO("yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


while True:
    success, img = cap.read()
    results = model(img, stream=True)
    # coordinates
    for r in results:
    
        boxes = r.boxes
        violate = []

        for box in boxes:
            cls = box.cls.tolist()[0]
            if cls == 0:
               x, y, w, h = box.xywh.tolist()[0]
               class_id = box.cls.tolist()[0]
               violate.append({'centroid': [x,y], 'violate': False})
        
        dist = []
        for i in range(0, len(violate)):
            pt1 = violate[i]['centroid']
            for j in range(i+1, len(violate)):
                pt2 = violate[j]['centroid']
                d = measure_distance(pt1, pt2)
                dist.append(d)     
            if min(dist, default=10000) < 400:
                violate[i]['violate'] = True
            else:
                violate[i]['violate'] = False
    
        for box in boxes:
            cls = box.cls.tolist()[0]
            if cls == 0:
            # bounding box
               x1, y1, x2, y2 = box.xyxy[0]
               x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            
               x,y, w,h = box.xywh.tolist()[0]
               centre = [x,y]         
       
               centroid = [cn['centroid'] for cn in violate]

               if centre in centroid:
                  idx = centroid.index(centre)
                  if violate[idx]['violate']:
                      color = (0, 0, 255)
                  else:
                      color=(0, 255, 0)
               cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

             # confidence
               confidence = math.ceil((box.conf[0]*100))/100
               print("Confidence --->",confidence)

            # class name
               cls = int(box.cls[0])
               print("Class name -->", classNames[cls])

            # object details
               org = [x1, y1]
               font = cv2.FONT_HERSHEY_SIMPLEX
               fontScale = 1
               color = (255, 0, 0)
               thickness = 2

               cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()