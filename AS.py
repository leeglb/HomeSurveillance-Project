import cv2

from ultralytics import YOLO
from config import API_KEY
from config import USER_EMAIL
from courier import Courier
from datetime import datetime 
from collections import deque
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
frameRate = (video_path.get(cv2.CAP_PROP_FPS))


fourccCode = cv2.VideoWriter_fourcc(*'mp4v')

recordedVideo = None 

videoDimensions = (frameWidth, frameHeight)
videoFileName = f"Video_Recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

BUFFER_SECONDS = 30
POST_EVENT_SECONDS = BUFFER_SECONDS #always equal 

BUFFER_SIZE = BUFFER_SECONDS * int(frameRate)
POST_EVENT_SIZE = BUFFER_SIZE # equal too 

frame_buffer = deque(maxlen=BUFFER_SIZE)
post_event_timer = 0 

system_recording = False
people_counter = 0
intrusion_time = 0.0 
end_time = 0.0 

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

                frame_buffer.append(annotated_frame)
                
                global start_capturing

                cv2.putText(annotated_frame, "Monitoring System Is Now Activated", (100, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                    

                """
                

                We have the following steps we need to ensure that is working: 

                1. The counter for intruders is on.
                2. The timer for any appearance of an intruder is on. (If exceeding 5 seconds, system alerts user).
                3. When the timer does exceed 5 seconds and intruders dissapear, the system will capture from when alarm is activated to 10 
                    seconds after no intruders are detected.
                
                """

                global people_counter
                global system_recording
                global intrusion_time
                global recordedVideo
                global post_event_timer
    
                people_counter = len(results[0].boxes) #counts number of boxes identified -> aka, number of intruders 

                cv2.putText(annotated_frame, f"Number of intruders detected: {people_counter}", (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                if people_counter > 0 and not system_recording: 

                    intrusion_time = time.time()

                    hours = int(intrusion_time // 3600)
                    minutes = int((intrusion_time % 3600) // 60)
                    seconds = int(intrusion_time % 60)
                    
                    cv2.putText(annotated_frame, f"Intruders have been present for: {hours:02d}:{minutes:02d}:{seconds:02d}.", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                    recordedVideo = cv2.VideoWriter(videoFileName,
                            fourccCode,
                            frameRate,
                            videoDimensions) # this activates the video writer --> starts to capture. 
                    
                    system_recording = True 
                    
                    for buffer_frame in frame_buffer: 

                        recordedVideo.write(buffer_frame) # append the previous frames. 

                    post_event_timer = POST_EVENT_SECONDS # now we record 30 seconds after. 

                if people_counter == 0 and system_recording:

                    recordedVideo.write(annotated_frame)
                    post_event_timer -= 1
                    
                    intrusion_time = 0.0

                    cv2.putText(annotated_frame, f"Intruders are not present.", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                    if post_event_timer <= 0: 

                        recordedVideo.release()

            
                cv2.putText(annotated_frame, str(current_datetime), (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
                cv2.imshow(WINDOW_NAME, annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                
                if key == ord("q"):
                    
                    break
                    
                elif key == ord("e"):
                    
                    self.email_system()
                                        
            else:
                
                break
        

        video_path.release()
        cv2.destroyAllWindows()

        



# Running functions. 
main = LiveFeed()
main.main_function()
