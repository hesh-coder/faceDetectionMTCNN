from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import cv2
import argparse

def drawBoxesRoundFaces(pixels, result_list, filename):
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    ax = pyplot.gca()
    
    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
        
        for key, value in result['keypoints'].items():
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)

    pyplot.show()

def drawJustFaces(pixels, result_list, filename):
    data = pyplot.imread(filename)

    for i in range(len(result_list)):
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height

        pyplot.subplot(1, len(result_list), i+1)
        pyplot.axis('off')

        pyplot.imshow(data[y1:y2, x1:x2])

    pyplot.show()

def beginDetection(pixels, outputType, filename):
    detector = MTCNN()
    faces = detector.detect_faces(pixels)

    if outputType == "f":
        drawJustFaces(pixels, faces, filename)
    elif outputType == "b":
        drawBoxesRoundFaces(pixels, faces, filename)
    else:
        print("Invalid output argument.")

parser = argparse.ArgumentParser(description="Detect faces in any given image file, choose output to either be highlighting on the face and features or cut out images of each face detected.")

parser.add_argument("-i", help="Input - path to the image file to be read.")

parser.add_argument("-o", help="Output - weather output should be boxes or faces (input arg should be 'b' or 'f'.")

args = parser.parse_args()

filename = args.i
outputType = args.o.lower()

#pixels = pyplot.imread(filename)
pixels = cv2.imread(filename)

beginDetection(pixels, outputType, filename)

## example input to draw detected faces on their own:
## python3 faceDetection_mtcnn.py -i test1.jpeg -o f
## example imput to draw whole images with faces and features marked:
## python3 faceDetection_mtcnn.py -i test1.jpeg -o b
