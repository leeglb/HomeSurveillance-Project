import cv2

from ultralytics import YOLO
from config import API_KEY
from config import USER_EMAIL
from courier import Courier
from datetime import datetime 
from collections import deque
import time 



class LiveFeed():
    
    def __init__(self):
        
        # API Key & ML Model -------------------------

        self.client = Courier(api_key = API_KEY)

        self.model = YOLO("yolo11n.pt") #pretrained model 

        # Video Path & Parameters --------------------

        self.video_path = cv2.VideoCapture(0)

        self.video_path.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_path.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.frameWidth = int(self.video_path.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.video_path.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frameRate =  (self.video_path.get(cv2.CAP_PROP_FPS))

        self.fourccCode = cv2.VideoWriter_fourcc(*'mp4v')

        self.recordedVideo = None 

        self.videoDimensions = (self.frameWidth, self.frameHeight)
        self.videoFileName = ""


        
        # Buffer Queue Parameters ------------------

        self.BUFFER_SECONDS = 15
        self.POST_EVENT_SECONDS = 15 #always equal 

        self.BUFFER_SIZE = self.BUFFER_SECONDS * int(self.frameRate)
        self.POST_EVENT_SIZE = self.BUFFER_SIZE # equal too 

        self.frame_buffer = deque(maxlen=self.BUFFER_SIZE)


        # Timers & Countdowns ---------------------

        self.system_recording = False

        self.intruder_count = 0

        self.intrusion_time = 0.0 

        self.end_time = 0.0 

        self.DURATION = 10

        self.EMAIL_COOLDOWN = 10  # subject to change depending on real world application (i.e 150-300)

        self.countdown = 0

        self.post_event_countdown = 0

        self.post_event_flag = False 
        self.post_event_timer = 0.0

        self.previous_intruder_count = 0 

        
        # System Boolean | Window Name | Final Frame 

        self.result_frame = None

        self.system_activated = False 

        self.first_presence = False # First presence detected. --> Kicks the system into alert. 

        self.email_sent = False

        self.WINDOW_NAME = "Monitoring System" 

        # ------------------------------------------

    def email_system(self):
        
        now = datetime.now()
            
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

        intruder_plural = "Intruder" if self.intruder_count == 1 else "Intruders"

        response = self.client.send.message(
            message = { 
            "to": { 
                "email": USER_EMAIL
            },

            "content": { 
                "title": "Home Surveillance Alert",
                "body": f"The System Has Detected {self.intruder_count} {intruder_plural} At {current_datetime}"
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
            
            boolean_ret, capture_frame = self.video_path.read()
            
            if(boolean_ret): # if successful 

                self.result_frame = capture_frame # result frame is the final frame we will display. 

                if self.system_activated: 

                    time_remaining = int(self.countdown - time.time())

                    if time_remaining >= 0: # system is now counting down towards activation.  

                        cv2.putText(capture_frame, f"Countdown to alarm activation: {time_remaining}", (20, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2) 

                    if time_remaining < 0: # not <= since both cv2.putText will display at 0 seconds. 
                    
                        results = self.model.track(capture_frame, persist=True, classes=0) #classes = 0 for just people tracking. 
                        
                        annotated_frame = results[0].plot() 

                        self.result_frame = annotated_frame

                        self.frame_buffer.append(annotated_frame)

                        cv2.putText(annotated_frame, "Monitoring System Is Now Activated", (20, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            
                        self.intruder_count = len(results[0].boxes) #counts number of boxes identified -> aka, number of intruders 
                        
                        cv2.putText(annotated_frame, f"Number of intruders detected: {self.intruder_count}", (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                        if (self.intruder_count > 0) and (self.system_recording == False): 

                            self.previous_intruder_count = self.intruder_count

                            self.first_presence = True 
                            
                            self.post_event_flag = False 

                            #intrusion_time = time.time()

                            #hours = int(intrusion_time // 3600)
                            #minutes = int((intrusion_time % 3600) // 60)
                            #seconds = int(intrusion_time % 60)
                            
                            #cv2.putText(annotated_frame, f"Intruders have been present for: {hours:02d}:{minutes:02d}:{seconds:02d}.", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                            self.videoFileName = f"Video_Recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

                            self.recordedVideo = cv2.VideoWriter(self.videoFileName,
                                    self.fourccCode,
                                    self.frameRate,
                                    self.videoDimensions) # this activates the video writer --> starts to capture. 
                            
                            self.system_recording = True 
                            
                            for buffer_frame in self.frame_buffer: 

                                self.recordedVideo.write(buffer_frame) # append the previous frames. 

                            
                            # File Writing ---------
                            
                            """
                            The file writing aspect of our system follows the following:
                            
                            1. Everytime an intruder is detected, it will alert the system
                                and log in a seperate file. 
                                
                            2. Due to the current system implementation, this will alert once for everytime a person appears
                                or dissapears from the system view.
                                
                            3. Depending on the intruder / intruders, this will determine the file content that we write. 
                            """
                            
                            file_path = "logbook.txt"
                            
                            file_content = ""
                            
                            if self.intruder_count == 1: 
                                
                                file_content = f"1 Intruder Detected At {current_datetime}.\n"
                                
                                with open(file_path, 'a') as file: 
                                
                                    file.write(file_content)
                                
                            elif self.intruder_count > 1: 
                            
                                file_content = f"{self.intruder_count} Intruders Detected At {current_datetime}.\n"
                                
                                with open(file_path, 'a') as file: 
                                    
                                    file.write(file_content)
                                    
                        
                        intruder_not_present = self.intruder_count == 0 and self.previous_intruder_count > 0
                        # intruder_currently_present = self.intruder_count > 0 and self.previous_intruder_count == 0

                        cv2.putText(annotated_frame, f"Intruders not present: " + str(intruder_not_present), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        cv2.putText(annotated_frame, "Previous intruder count: " + str(self.previous_intruder_count), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                                    
                        if intruder_not_present and not self.post_event_flag:
                            
                            self.post_event_flag = True 
                            
                            self.post_event_timer = time.time()
                            
                            cv2.putText(annotated_frame, f"Intruders are not present.", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                            
                        if self.post_event_flag:
                            
                            self.recordedVideo.write(annotated_frame) 
                            
                            self.post_event_countdown = int(time.time() - self.post_event_timer)
                            
                            time_since_dissappearance = self.POST_EVENT_SECONDS - self.post_event_countdown
                            
                            cv2.putText(annotated_frame, str(time_since_dissappearance), (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                            
                            if time_since_dissappearance <= 0: 
                                
                                self.post_event_flag = False 
                                
                                self.system_recording = False 

                                self.email_sent = False 

                                self.first_presence = False 

                                self.previous_intruder_count = 0

                                 # Email Sending -------
                            
                                """
                                1.For our email system, we have already created the email function above the main function.
                                
                                2. The email system will activate only if an intruder is detected. As a result, we follow 
                                the current if loop, allowing us to have a cooldown period. 
                                
                                3. If an intruder constantly appears, this will cause the system to spam ping the user.
                                
                                4. To avoid this, we allow the system to wait for the post even timer to finish.
                                
                                5. When the email is sent, we will also send a log to the logbook to record intrusion alerts. 
                                
                                """

                                self.email_system()

                                self.email_sent = True 

                                with open(file_path, 'a') as file: 
                                    
                                        file.write(f"Alert Sent At {current_datetime}.\n")
                                
                                self.recordedVideo.release()

                                self.frame_buffer.clear()
                
                                

    
            
                cv2.putText(self.result_frame, str(current_datetime), (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
                cv2.imshow(self.WINDOW_NAME, self.result_frame)

                key = cv2.waitKey(1) & 0xFF
                
                if key == ord("q"):
                    
                    break

                elif key == ord("s"):
                    
                    if self.system_activated:
                        
                        pass

                    else:
                        
                        self.countdown = time.time() + self.DURATION

                        self.system_activated = True 
                        
                                        
            else:
                
                break
            

        self.video_path.release()
        cv2.destroyAllWindows()

        



# Running functions. 
main = LiveFeed()
main.main_function()