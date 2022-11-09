from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import cv2


def compare_two_images():
    # They are different
    image1 = np.array(Image.open("image.png"))
    image2 = np.array(Image.open("test/data/RL_20210304_0.jpg"))
    print("hhh")

if __name__ == "__main__":
    
    plot = np.array(Image.open("res_plot.png"))
    
    image = Image.open("res_render.png")
    
    invert_image = ImageOps.invert(image)
    im = np.array(image)
    invert_im = np.array(invert_image)
    
    x, y, _ = np.where(invert_im)
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    
    crop = im[min_x:max_x, min_y:max_y, :]
    
    # imageBox = invert_im.getbbox()
    # imageBox = tuple(np.asarray(imageBox))
    
    # cropped=image.crop(imageBox)
    
    # # np.argwhere(np.array(image))
    
    # print(image)
    
    
    # ret,thresh = cv2.threshold(image,0,127,cv2.THRESH_BINARY)
    # white_pt_coords=np.argwhere(thresh)
    # min_y = min(white_pt_coords[:,0])
    # min_x = min(white_pt_coords[:,1])
    # max_y = max(white_pt_coords[:,0])
    # max_x = max(white_pt_coords[:,1])
    
    # crop = image[min_y:max_y,min_x:max_x]
    # plt.imshow(crop)
    # plt.show()
    
    # x, y, z = np.where(image)
    
    # min_x, max_x = min(x), max(x)
    # min_y, max_y = min(y), max(y)
    # min_z, max_z = min(z), max(z)
    
    # filepath = "res_render.png"
    
    # image=Image.open(filepath)
    # image.load()
    # imageSize = image.size

    # # remove alpha channel
    # invert_im = image.convert("RGB")

    # # invert image (so that white is 0)
    # invert_im = ImageOps.invert(invert_im)
    # imageBox = invert_im.getbbox()
    # imageBox = tuple(np.asarray(imageBox)+0)

    # cropped=image.crop(imageBox)
    # print(filepath, "Size:", imageSize, "New Size:", imageBox)
    # cropped.save(filepath)