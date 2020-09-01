import os
import PIL.Image as Image

ImageSize = 110
Version = "0.1.6.2"
input_path = "D:\\jonod\\Pictures\\BackpropExperiment" # path to input folder
output_path = "D:\\jonod\\Pictures\\BackpropExperiment\\Outputs" # path to output folder

# Get all filenames in path
f = []
for (_, _, filenames) in os.walk(input_path):
    f.extend(filenames)
    break

imageCount = len(f)
interval = 1/imageCount
size = (ImageSize, ImageSize)
output_file = open(output_path + "\\" + Version + "_training.txt", "w")
count = 0
for num in range(0, imageCount):
    im = Image.open(input_path + "\\" + f[num])
    resized = im.resize(size)
    im = resized
    output_file.write(f[num])
    for num2 in range(0, 8):
        if count != num2:
            output_file.write("|0")
        else:
            output_file.write("|1")
    for x in range(0, ImageSize):
        for y in range(0, ImageSize):
            try:
                pixVal = im.getpixel((x, y))
                output_file.write("|" + str(float(pixVal[0] + float(pixVal[1]) + float(pixVal[2]))/(3.0 *255.0)))
            except:  # This is bad practice
                output_file.write("|0.0")  # If it is outside of the range of the thumbnail it should be transparent
    if num != imageCount-1:
        output_file.write("!")
    count += 1
output_file.close()