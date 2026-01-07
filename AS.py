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

start_capturing = False 

frameWidth = int(video_path.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video_path.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameRate = int(video_path.get(cv2.CAP_PROP_FPS))

if frameRate == 0 or frameRate is None:

    frameRate = 30

frameRate = float(frameRate)

fourccCode = cv2.VideoWriter_fourcc(*'mp4v')

videoDimensions = (frameWidth, frameHeight)
videoFileName = f"Video_Recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

recordedVideo = cv2.VideoWriter(videoFileName,
                            fourccCode,
                            frameRate,
                            videoDimensions)

people_counter = 0
intrusion_length = 0.0 

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
                
                annotated_frame = results[0].plot() 
                
                global alarm_started
                global countdown
                global start_capturing
                 
                if alarm_started: 

                    time_remaining = int(countdown - time.time())

                    if time_remaining > 0: 

                        cv2.putText(annotated_frame, f"Countdown To Alarm Activation: {time_remaining}", (20, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                        
                    if time_remaining <= 0: # Alarm is now activated 

                        """
                        For this part, we are essentially saying the alarm is activated.

                        We have the following steps we need to ensure that is working: 

                        1. The counter for intruders is on.
                        2. The timer for any appearance of an intruder is on. (If exceeding 5 seconds, system alerts user).
                        3. When the timer does exceed 5 seconds and intruders dissapear, the system will capture from when alarm is activated to 10 
                           seconds after no intruders are detected.
                        
                        """

                        global people_counter
                        
                        cv2.putText(annotated_frame, "Alarm Is Activated", (20, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            
                        people_counter = len(results[0].boxes) #counts number of boxes identified -> aka, number of intruders 

                        cv2.putText(annotated_frame, f"Number of intruders detected: {people_counter}", (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                        if people_counter > 0: 

                            start_time = datetime.now().second

                            cv2.putText(annotated_frame, f"Intruders have been present for: {start_time} seconds.", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                            if people_counter == 0:

                                end_time = time.time()

            
                            
                
                cv2.putText(annotated_frame, str(current_datetime), (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                recordedVideo.write(annotated_frame)
                
                cv2.imshow(WINDOW_NAME, annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                
                if key == ord("q"):
                    
                    break

                elif key == ord("s"): #start alarm 
                    
                    alarm_started = True
                    
                    countdown = time.time() + DURATION
                    
                elif key == ord("e") and alarm_started:
                    
                    self.email_system()

                elif key == ord("c"):

                    recordedVideo.release() # release means we stop recording. 
                                        
            else:
                
                break
        

        video_path.release()
        cv2.destroyAllWindows()

        



# Running functions. 
main = LiveFeed()
main.main_function()