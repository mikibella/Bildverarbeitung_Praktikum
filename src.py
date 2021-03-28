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
    array = np.array([[grayValue for i in range(width)]for j in range(height)])
    print(array)

    # show image with matplotlib as grayscale image
    plt.imshow(array, cmap='gray', vmin=0, vmax=255)
    plt.show()

    # Bild einlesen
    #img = Image.open(PATH)

    # Bild speicher
    # img.save(PATH)
