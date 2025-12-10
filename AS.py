import cv2

from ultralytics import YOLO
from config import API_KEY
from config import USER_EMAIL
from courier import Courier

# Global Variables -------------------------

client = Courier(api_key = API_KEY)

print(API_KEY)

model = YOLO("yolo11n.pt") #pretrained model 

video_path = cv2.VideoCapture(0)

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
                "body": "The System Has Detected Intruders"
            },

            "routing": {
                "method": "single",
                "channels": ["email"]
            }
            }
        )
        print(response.request_id)
        print(response)
        

    def main_function(self):

        while(True): 
            
            boolean_ret, capture_frame = video_path.read()
            
            if(boolean_ret): # if successful 
                
                results = model.track(capture_frame, persist=True, classes=0) #classes = 0 for just people tracking. 
                
                annotated_frame = results[0].plot() 
                
                cv2.imshow("Monitoring System", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                
                if key == ord("q"):
                    
                    break

                elif key == ord("s"):

                    self.email_system()
                
            else:
                
                break
        
        
        video_path.release()
        cv2.destroyAllWindows()


# Running functions. 
main = LiveFeed()
main.main_function()