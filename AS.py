import cv2

from ultralytics import YOLO
from config import API_KEY
from config import USER_EMAIL
from courier import Courier
from datetime import datetime 
import time 


# Global Variables -------------------------

client = Courier(api_key = API_KEY)

model = YOLO("yolo11n.pt") #pretrained model 

video_path = cv2.VideoCapture(0)

video_path.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_path.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

people_counter = 0 

alarm_started = False

DURATION = 10

countdown = 0.0

WINDOW_NAME = "Monitoring System" 
# ------------------------------------------


class LiveFeed():

    def email_system(self):

        response = client.send.message(
            message = { 
            "to": { 
                "email": USER_EMAIL
            },

            "content": { 
                "title": "Home Surveillance Alert",
                "body": f"The System Has Detected {people_counter} Intruders"
            },

            "routing": {
                "method": "single",
                "channels": ["email"]
            }
            }
        )

    def main_function(self):

        while(True): 
            
            current_datetime = datetime.now()
            
            boolean_ret, capture_frame = video_path.read()
            
            if(boolean_ret): # if successful 
                
                results = model.track(capture_frame, persist=True, classes=0) #classes = 0 for just people tracking. 
                
                people_counter = len(results[0].boxes) #counts number of boxes identified
                
                annotated_frame = results[0].plot() 
                
                global alarm_started
                global countdown
                 
                if alarm_started: 

                    time_remaining = int(countdown - time.time())

                    if time_remaining >= 0: 

                        cv2.putText(annotated_frame, f"Countdown To Alarm Activation: {time_remaining}", (20, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                        
                        if time_remaining <= 0: 
                            
                            if people_counter >= 1:
                                
                                self.email_system()
                
                cv2.putText(annotated_frame, str(current_datetime), (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
                cv2.putText(annotated_frame, f"Number of individuals detected: {people_counter}", (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
                cv2.imshow(WINDOW_NAME, annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                
                if key == ord("q"):
                    
                    break

                elif key == ord("s"): #start alarm 
                    
                    alarm_started = True
                    
                    countdown = time.time() + DURATION

                                        
            else:
                
                break
        
        
        video_path.release()
        cv2.destroyAllWindows()
        



# Running functions. 
main = LiveFeed()
main.main_function()