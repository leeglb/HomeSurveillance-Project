<h1> Surveillance System - Using Object Detection </h1>

<h2> The aim of the project: </h2>

The aim of this project is to create a monitoring system
for home surveillance with the capabilities of direct, live contact
when intruders are detected.

In the day and age where machine learning and artifical intelligence
is utilised to track & monitor for targetted individuals and tracking,
it is pivotal to understand the key technology and current software 
for this sector. 

<h3> Prerequisites (Capturing System File): </h3>

1. Ultralytics: YOLO (Tracking Software)

```
pip install -U ultralytics
```

2. Courier: (Email / SMS Service)

<i> Link is provided for setup </i>: https://www.courier.com/

``` 
pip install trycourier
```

3. OpenCV: (Capturing System)

```
pip install opencv-contrib-python

```

4. Some modules to import into our main file:

```
from datetime import datetime
import time
```

5. dotenv

``` 
pip install dotenv
```


<h3> Prerequisites (Jupyter Notebook File): </h3>

1. Pandas:

```
pip install pandas
```

2. Matplotlib:

```
pip install Matplotlib
```

3. StatsModels:

``` 
pip install -U statsmodels==0.14.4 scipy==1.14.1
```

<h3> Config File & .env File: <h3>

1. Create a .env file which will contain both:

```
COURIER_API_KEY = 'YOUR_API_KEY'

EMAIL = 'YOUR_EMAIL_HERE'
```

Essentially, your .env file is a plain text file utilised to store our environment variables (particularly API keys), 
seperate from our application source code. (Hides important info pretty much.)

2. Create a config.py file (In Source Code Already):

<i> config.py <i>

```
from dotenv import load_dotenv
import os 

load_dotenv()

API_KEY = os.getenv("COURIER_API_KEY")
USER_EMAIL = os.getenv("EMAIL")
```

<h3> Why am I making this project? <h3>

Whilst the project is nothing over the top, the importance of learning how to utilise prexisting tools,
particularly in the field of machine learning and camera operation is important. OpenCV is an extensive tool for image and 
camera tasks, in essense a really fun and interactive library to learn.

Additionally, I wanted to learn simple, yet crucial files and file management within a project. Using config files, .env, .gitignore
are all basic, but a must when creating a public project and for ease of access with classified variables. 

Furthermore, whilst I could have created my own or trained my own machine learning model (may do for the future), I find that given
my current circumstance, why reinvent the wheel, when I can use the wheel to create something else. We won't create our own AI 
just to help us to debug or do homework (unless we wanted to learn how to actually create one ourselves and understand the true
nature and background). No shame to this, but with my timeframe it is hard to replicate. 

To make the project stand out a bit more, I added the knowledge I learnt from my university course (COMP2011), utilising an ARIMA 
model to predict the crime rate within Australia which helped to strengthen my reasoning for my product choice. It is a good way to 
storytell and be interactive with what we are making. 