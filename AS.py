import cv2

from ultralytics import YOLO
from config import API_KEY
from config import USER_EMAIL
from courier import Courier
from datetime import datetime 
from collections import deque
import time 





# API Key & ML Model -------------------------

client = Courier(api_key = API_KEY)

model = YOLO("yolo11n.pt") #pretrained model 




# Video Path & Parameters --------------------

video_path = cv2.VideoCapture(0)

video_path.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_path.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

start_capturing = False 

frameWidth = int(video_path.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video_path.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameRate =  (video_path.get(cv2.CAP_PROP_FPS))

fourccCode = cv2.VideoWriter_fourcc(*'mp4v')

recordedVideo = None 

videoDimensions = (frameWidth, frameHeight)
videoFileName = f"Video_Recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"





# Buffer Queue Parameters ------------------

BUFFER_SECONDS = 30
POST_EVENT_SECONDS = BUFFER_SECONDS #always equal 

BUFFER_SIZE = BUFFER_SECONDS * int(frameRate)
POST_EVENT_SIZE = BUFFER_SIZE # equal too 

frame_buffer = deque(maxlen=BUFFER_SIZE)
post_event_timer = 10 




# Timers & Countdowns ---------------------

system_recording = False
intruder_count = 0
intrusion_time = 0.0 
end_time = 0.0 

DURATION = 10

countdown = 0




# Email Variables -------------------------

email_cooldown = 10 # subject to change depending on real world application (i.e 150-300)
email_sent = False




# System Boolean | Window Name | Final Frame 

result_frame = None

system_activated = False 

WINDOW_NAME = "Monitoring System" 

# ------------------------------------------


class LiveFeed():

    def email_system(self):
        
        now = datetime.now()
            
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

        response = client.send.message(
            message = { 
            "to": { 
                "email": USER_EMAIL
            },

            "content": { 
                "title": "Home Surveillance Alert",
                "body": f"The System Has Detected {intruder_count} Intruders At {current_datetime}"
            },

            "routing": {
                "method": "single",
                "channels": ["email"]
            }
            }
        )

    def main_function(self):

        while(True): 
            
            now = datetime.now()
            
            current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
            
            boolean_ret, capture_frame = video_path.read()
            
            if(boolean_ret): # if successful 


                """
                1. We want to install a countdown, a time for the user to be able to leave the house.
                2. In this time before the countdown concludes, we will utilise the pre-annotated frame.
                3. Present the date, time but no detection, no intruder count. 

                """

                global countdown
                global system_activated
                global result_frame

                result_frame = capture_frame # result frame is the final frame we will display. 

                if system_activated: 

                    time_remaining = int(countdown - time.time())

                    if time_remaining >= 0: # system is now counting down towards activation.  

                        cv2.putText(capture_frame, f"Countdown to alarm activation: {time_remaining}", (20, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2) 

                    if time_remaining <= 0: 

                        global start_capturing
                    
                        results = model.track(capture_frame, persist=True, classes=0) #classes = 0 for just people tracking. 
                        
                        annotated_frame = results[0].plot() 

                        result_frame = annotated_frame

                        frame_buffer.append(annotated_frame)

                        cv2.putText(annotated_frame, "Monitoring System Is Now Activated", (20, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                            

                        """
                        We have the following steps we need to ensure that is working: 

                        1. The counter for intruders is on.
                        2. The timer for any appearance of an intruder is on. (If exceeding 5 seconds, system alerts user).
                        3. When the timer does exceed 5 seconds and intruders dissapear, the system will capture from 
                            when alarm is activated to 10 seconds after no intruders are detected.
                        """

                        global intruder_count
                        global system_recording
                        global intrusion_time
                        global recordedVideo
                        global post_event_timer
                        global email_sent
            
                        intruder_count = len(results[0].boxes) #counts number of boxes identified -> aka, number of intruders 

                        cv2.putText(annotated_frame, f"Number of intruders detected: {intruder_count}", (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                        if intruder_count > 0 and not system_recording: 

                            #intrusion_time = time.time()

                            #hours = int(intrusion_time // 3600)
                            #minutes = int((intrusion_time % 3600) // 60)
                            #seconds = int(intrusion_time % 60)
                            
                            #cv2.putText(annotated_frame, f"Intruders have been present for: {hours:02d}:{minutes:02d}:{seconds:02d}.", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                            recordedVideo = cv2.VideoWriter(videoFileName,
                                    fourccCode,
                                    frameRate,
                                    videoDimensions) # this activates the video writer --> starts to capture. 
                            
                            system_recording = True 
                            
                            for buffer_frame in frame_buffer: 

                                recordedVideo.write(buffer_frame) # append the previous frames. 
                                
                            email_time = int((email_cooldown + time.time()) - time.time())

                            if email_time <= 0 and not(email_sent):

                                self.email_system()

                                email_sent = True 


                            
                            # File Writing ---------
                            
                            file_path = "logbook.txt"
                            
                            file_content = ""
                            
                            if intruder_count == 1: 
                                
                                file_content = f"1 Intruder Detected At {current_datetime}.\n"
                                
                                with open(file_path, 'a') as file: 
                                
                                    file.write(file_content)
                                
                            elif intruder_count > 1: 
                            
                                file_content = f"{intruder_count} Intruders Detected At {current_datetime}.\n"
                                
                                with open(file_path, 'a') as file: 
                                    
                                    file.write(file_content)

                        if system_recording:

                            recordedVideo.write(annotated_frame)

                            if intruder_count == 0: 
                                
                                post_event_timer = int(10 - time.time())
                                
                                if post_event_timer == 0: 
                                    
                                    recordedVideo.release()
                                    
                                    system_recording = False # no longer capturing. 
                                
                                cv2.putText(annotated_frame, f"Intruders are not present.", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                                
                                cv2.putText(result_frame, str(current_datetime), (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
                cv2.putText(result_frame, str(current_datetime), (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
                cv2.imshow(WINDOW_NAME, result_frame)

                key = cv2.waitKey(1) & 0xFF
                
                if key == ord("q"):
                    
                    break

                elif key == ord("s"):

                    countdown = time.time() + DURATION

                    system_activated = True 
                    
                elif key == ord("e"):
                    
                    self.email_system()
                                        
            else:
                
                break
        

        video_path.release()
        cv2.destroyAllWindows()

        



# Running functions. 
main = LiveFeed()
main.main_function()
