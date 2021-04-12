# Recognize 2000+ faces with your Jetson Nano.
![output image]( https://qengineering.eu/images/John_Cleese.png )

## A fast face recognition and face recording running on a Jetson Nano.
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)<br/>

This C++ application recognizes a person from a database of more than 2000 faces.  It is built for a Jetson Nano, but can easily be ported to other platforms.

First, the faces and their landmarks are detected by **RetinaFace** or **MTCNN**.
Next, the database is scanned with **Arcface** for the matching face.
In the end, **Face Anti Spoofing** tests whether the person in front of the camera is real and not a mask or a cardboard photo.

If the face is not found in the database, it will be **added automatically**. A **blur** filter ensures only sharp faces in the database. One photo per person is sufficient, although more does not hurt.

Special made for a Jetson Nano see [Q-engineering deep learning examples](https://qengineering.eu/deep-learning-examples-on-raspberry-32-64-os.html) <br/>

------------

## Benchmark.
| Model  | Jetson Nano 2015 MHz | Jetson Nano 1479 MHz | RPi 4 64-OS 1950 MHz | RPi 4 64-OS 1500 MHz |
| ------------- | :------------: | :-------------: | :-------------:  | :-------------: |
| MTCNN  | 11 mS | 14 mS  | 22 mS | 25 mS  |
| RetinaFace  | 15 mS  | 19 mS  | 35 mS  | 37 mS  |
| ArcFace  | +17 mS | +21 mS  | +36 mS  | +40 mS  |
| Spoofing | +25 mS  | +37 mS  | +37 mS  | +45 mS  |


------------

## Dependencies.
### April 4 2021: Adapted for ncnn version 20210322
To run the application, you have to:
- The Tencent ncnn framework installed. [Install ncnn](https://qengineering.eu/install-ncnn-on-jetson-nano.html) <br/>
- Code::Blocks installed. (`$ sudo apt-get install codeblocks`)

------------

## Installing the app.
To extract and run the application in Code::Blocks <br/>
$ mkdir *MyDir* <br/>
$ cd *MyDir* <br/>
$ wget https://github.com/Qengineering/Face-Recognition-Jetson-Nano/archive/refs/heads/master.zip <br/>
$ unzip -j master.zip <br/>
Remove master.zip and README.md as they are no longer needed. <br/> 
$ rm master.zip <br/>
$ rm README.md <br/> <br/>
Your *MyDir* folder must now look like this: <br/> 
Graham Norton.jpg (example image)<br/>
FaceRecognition.cbp (code::blocks project file) <br/>
Norton_A.mp4 (movie with faces to load) <br/>
Norton_2.mp4 (movie to check)<br/>
*img* (database folder) <br/>
*models* (folder with used ncnn deep learning models) <br/>
*src* (C++ source files)<br/>
*include* (the C++ headers)<br/>

------------

## Running the app.
To run the application load the project file FaceRecognition.cbp in Code::Blocks.<br/> 
First, we are going to fill the database with new faces. The database *img*  initial holds one face, Graham.jpg.<br/><br/>
![output image]( https://qengineering.eu/images/Strangers1.png )<br/><br/>
Check in main.cpp line 253. It must be `cv::VideoCapture cap("Norton_A.mp4");` <br/>

Compile and run the app. Movie *Norton_A.mp4* will be played and new faces are stored in the database. In the end, you have the database filled as below.<br/><br/>
![output image]( https://qengineering.eu/images/Strangers2.png )<br/><br/>

Next, alter the name of the movie in line 253 of main.cpp to *Norton_2.mpg*.
Compile and run the application again. You will see that all the faces are correctly recognized. It can still happen that faces are added to the database due to strange angles or grimaces.<br/>

------------

## Database.
The application can easily contain more than 2000 faces. There are reports that ArcFace works flawlessly with over 5000 faces. With large databases, it is important to keep your face "natural". It means a front view photo with eyes open and mouth closed without a smile or other funny faces.<br/>
The database is filled "on the fly", as you have seen above. It is also possible to manually add a face to the databases. To do this, run the application from the command-line and enter the name of the image as an argument. For example `./FaceRecognition "Graham Norton.jpg"` Note the quotation marks around the name if it has a space.

You can give the faces a corresponding name. By using a hash, you can associate multiple pictures with the same name.<br/><br/>
![output image]( https://qengineering.eu/images/Strangers3.jpg )<br/><br/>
By the way, note the warp perspective of Graham Norton's face that we added via a command-line argument and the crop of the same photo already saved in the database. This is done by the ArcFace algorithms.

The **blur filter** prevents vague or imprecise faces from being added to the database. Below you see a few examples of faces we encounter in the database when de blur filter was switched off.<br/><br/>
![output image]( https://qengineering.eu/images/Strangers4.jpg )<br/><br/>
Another safety measure is the orientation of the face. Only faces in front of the camera are added to the database. Faces "in profile" are often inaccurate in large databases.<br/>

------------

## Code.
The application is written in C ++. The setup is flexible and easy to adapt to your own needs. See it as a skeleton which you can expand yourself. Some hints.
In main .cpp at line 21 you see a few defines.
```
#define RETINA                  //comment if you want to use MtCNN landmark detection instead
#define RECOGNIZE_FACE
#define TEST_LIVING
#define AUTO_FILL_DATABASE
#define BLUR_FILTER_STRANGER
// some diagnostics
#define SHOW_LEGEND
#define SHOW_LANDMARKS
```
By commenting the line the define is switched off. For instance, if you do not want to incorporate the **anti-spoofing** test (saves you 37 mS), comment this line. The MtCNN face detection is switched on by turning RETINA off.<br/>
Another important point is that only one face is labelled. It is no problem to loop through all faces. However, they are usually too small to be recognized with great accuracy. Besides, your FPS will drop also.
Note, the input image for the RetinaFace is 324 x 240 pixels. Larger pictures are resized to that format. ArcFace works with an input of 112 x 112 pixels. 
If you have a large input format, you could extract the faces at a larger scale from this image, once you have the coordinates from the RetinaFace network. Now, faces are to be recognized with much greater accuracy. Of course, there will be not much of an FPS left.

------------


## WebCam.
If you want to use a camera please alter line 253 in main.cpp to<br/>
`cv::VideoCapture cap(0);                          //WebCam`<br/>
If you want to run a movie please alter line 253 in main.cpp to<br/>
`cv::VideoCapture cap("Norton_2.mp4");   //Movie`<br/>

------------

## Papers.
[MTCNN](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)<br/>
[RetinaFace](https://arxiv.org/pdf/1905.00641.pdf)<br/>
[ArcFace](https://arxiv.org/pdf/1801.07698.pdf)<br/>
[Anti spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/README_EN.md)<br/>

------------

### Thanks.
https://github.com/Tencent/ncnn<br/>
https://github.com/nihui<br/>
https://github.com/LicheeX/MTCNN-NCNN<br/>
https://github.com/XinghaoChen9/LiveFaceReco_RaspberryPi<br/>
https://github.com/deepinsight/insightface<br/>
https://github.com/minivision-ai/Silent-Face-Anti-Spoofing <br/>
https://github.com/Qengineering/Blur-detection-with-FFT-in-C
