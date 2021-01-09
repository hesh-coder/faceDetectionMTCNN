# faceDetectionMTCNN

This program uses MTCNN (Multi-task Cascade Convolutional Neural Network). This is taken from the mtcnn library.\
I learnt how to do such detection with the following tutorial:\
https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/

(I did not create the MTCNN, it is a classifier that has been refined over the past 20 years)

Using mtcnn has means using the cascade structure on three networks:
1. Candidate windows are developed through a not too deep CNN (Cascade Neural Network)
2. Using a deeper CNN, lots of non-face windows are found and removed.
3. Finally a very deep and powerful CNN uses these results to output accurate facial feature positions (which are also known
as landmark positions).

My program works as follows:\
You pass it an image file tagged as -i (input) and output type tagged as -o (output).\
The output must be 'f' or 'b', where 'f' gives cut out faces detected and 'o' gives the original image with faces and features marked.

To detect all faces in 'test1.png' and output detected faces as isolated images:\
python3 faceDetection_mtcnn.py -i test1.png -o f\
<br></br>
To detect all faces and output the original image with faces and facial features marked:\
python3 faceDetection_mtcnn.py -i test1.png -o b
