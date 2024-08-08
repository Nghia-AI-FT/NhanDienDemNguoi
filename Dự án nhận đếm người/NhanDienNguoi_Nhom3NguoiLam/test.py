import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from tracker import*

model=YOLO('yolov8s.pt')
stream=cv2.VideoCapture('persons.mp4')
# stream = CamGear(source='https://www.youtube.com/watch?v=nt3D26lrkho', stream_mode = True, logging=True).start() # YouTube Video URL as input
# stream = CamGear(source='https://www.youtube.com/watch?v=9bFOCNOarrA', stream_mode = True, logging=True).start() # YouTube Video URL as input
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

classnames = []
counter_down=[]
counter_up=[]
down={}
up={}

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
count_up=0
count_down=0

tracker =Tracker()
#khai báo vị trí khi đối tượng đi qua đó sẽ nhận diện 
# area=[(497,240),(340,240),(70,360),(460,360)]

# area2=[(688,240),(532,240),(538,360),(857,360)]

# area=[(510,200),(270,200),(270,360),(510,360)]

# area2=[(730,200),(510,200),(510,360),(730,360)]

area=[(730,200),(270,200),(270,280),(730,280)]

area2=[(730,280),(270,280),(270,360),(730,360)]
while True:
    #đọc file vid dùng cái này    
    ret,frame = stream.read()
    if not ret:break  
    # #xài stream thì dùng cái này 
    # frame = stream.read()

    # tang toc do vid
    # count += 1
    # if count % 3 != 0:
    #     continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    print(px)
    list=[]
    
    for index,row in px.iterrows():
        
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        p=float(row[4])
        d=int(row[5])
        c=class_list[d]
        if 'person'  in c:
            list.append([x1,y1,x2,y2,p])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id,p=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        pp=p
         
        y=200
        z=360 
         
        offset = 7
    
           
        
        # nếu đối tượng đi qua vùng này thì thêm id nó vào biến down            
        result=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        
        if result>=0:
            down[id]=cy
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
            cvzone.putTextRect(frame,f'{c, (round(pp*100,2))}',(x3,y3),1,1)
         
        if id in down:
                if z < (cy + offset) and z > (cy - offset)and pp>0.5:  
                    if counter_down.count(id)==0:
                        counter_down.append(id)
        # nếu đối tượng đi qua vùng này thì thêm id nó vào biến up                    
        result2=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)                
        if result2>=0:
            up[id]=cy
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
            cvzone.putTextRect(frame,f'{c,round(pp*100,2)}',(x3,y3),1,1)   
        if id in up:
           if y < (cy + offset) and y > (cy - offset) and pp >0.5:         
             if counter_up.count(id)==0:
                counter_up.append(id)            
                # print(counter_down)
        
        
    text_color = (0,0,0)  # Black color for text
    black_color = (0,0,0)  # (B, G, R)green_color = (0, 255, 0)
    red_color = (0, 0, 255)  # (B, G, R)   
    blue_color = (255, 0, 0)  # (B, G, R) 

    
    cv2.line(frame,(270,y),(730,y),red_color,3)  #  starting cordinates and end of line cordinates
    cv2.putText(frame,('In line'),(205,200),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.line(frame,(270,z),(730,z),blue_color,3)  # seconde line
    cv2.putText(frame,('Out line'),(200,360),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)    
    
    
    cv2.polylines(frame,[np.array(area,np.int32)],True,(255,255,255),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)


    # downwards = (len(counter_down))
    # cv2.putText(frame,('go outside - ')+ str(downwards),(60,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 1, cv2.LINE_AA)    
    # downwardss = (len(counter_up))
    # cv2.putText(frame,('go into - ')+ str(downwardss),(60,60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 1, cv2.LINE_AA) 
    
    downwards = (len(counter_down))
    cv2.putText(frame,('go outside - ')+ str(downwards),(60,60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color, 1, cv2.LINE_AA)    
    downwardss = (len(counter_up))
    cv2.putText(frame,('go into - ')+ str(downwardss),(60,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color, 1, cv2.LINE_AA) 
    
    cv2.imshow("RGB", frame)


    if cv2.waitKey(1)&0xFF==27:
        break
stream.release()
cv2.destroyAllWindows()
