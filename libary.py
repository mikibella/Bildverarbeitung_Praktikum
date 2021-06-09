import numpy as np
from PIL import Image  # Libary for image loading saving and manipulation
 
class Libary:
       img = Image.Image()


       #Konstruktor um Attribute zu setzen
       def __init__(self,PATH):
              self.img =Image.open(PATH) 
              
       def convertGrayscale(self):
              img1 = self.img.convert('L') 
              newArray = np.asarray(img1) 
              return newArray
