import numpy as np
from PIL import Image 

class Manual:
    picArray = []  

    #Konstruktor um Attribute zu setzen
    def __init__(self,PATH):
        img = Image.open(PATH)
        self.array = np.asarray(img)


    def convertGrayscale(self):
        newArray = np.zeros(shape=(len(self.array),len(self.array[0])))
        #Durch das Bild iterrieren
        for x in range(len(self.array)):
            for y in range(len(self.array[0])):
                #Farbanteile der einzelnen pixel mit Vorfaktoren addieren
                newArray[x][y] = 0.2989 * self.array[x][y][0] + 0.5870 *  self.array[x][y][1] + 0.1140 * self.array[x][y][2]
        return newArray