from PIL import Image


import os

directory = r'../Stanford-cs230-final-project/clips/label_image/'
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        print(directory + filename)
        im = Image.open(directory + filename)
        pixelMap = im.load()
        img = Image.new( im.mode, im.size)
        pixelsNew = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if pixelMap[i,j] == (255, 255, 255, 255):
                    pixelsNew[i,j] = (1,1,1,255)
                else:
                    pixelsNew[i,j] = pixelMap[i,j]
        img.save(directory + filename)
        
