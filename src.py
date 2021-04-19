from PIL import Image  # Libary for image loading saving and manipulation
import matplotlib.pyplot as plt  # libary for showing image
import numpy as np  # numpy libary

if __name__ == "__main__":
    # Greetings to the World
    print("Moin World")

    # some local variables
    width = 800
    height = 800
    grayValue = 0  # black
    y = 0
    x = 0

    print("width = {}".format(width))
    print("height = {}".format(height))

    # creating 2D Array
    #array = np.array([[grayValue for i in range(width)]for j in range(height)])
    # print(array)

    # show image with matplotlib as grayscale image
    #plt.imshow(array, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    # Bild einlesen
    PATH = r"C:\Users\Truckin\Desktop\Studium\6.Semester\Bildverarbeitung\Verkehrsschilder\vfa_07.jpg"
    img = Image.open(PATH)
    array = np.asarray(img)
    newArray = np.zeros((len(array), len(array[0])))
    #print(newArray)
    gray=0
    #for x in range(len(array)):
    #    print(x)
    #    for y in range(len(array[0])):
    #        newArray[x][y] = 0.2989 * array[x][y][0] + 0.5870 *  array[x][y][1] + 0.1140 * array[x][y][2]
    img1 = img.convert('L') 
    newArray = np.asarray(img1) #convert a gray scale
    print(newArray)
    plt.imshow(newArray,cmap='gray', vmin=0, vmax=255)
    plt.show()
