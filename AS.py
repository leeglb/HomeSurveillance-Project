import cv2

from ultralytics import YOLO

model = YOLO("yolo11n.pt") #pretrained model 

video_path = cv2.VideoCapture(0)


while(True): 
    
    boolean_ret, capture_frame = video_path.read()
    
    if(boolean_ret): # if successful 
        
        results = model.track(capture_frame, persist=True)
        
        annotated_frame = results[0].plot() 
        
        cv2.imshow("Monitoring System", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            
            break
        
    else:
        
        break
        
        
video_path.release()
cv2.destroyAllWindows()