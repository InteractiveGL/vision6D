from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2


def compare_two_images():
    # They are different
    image1 = np.array(Image.open("image.png"))
    image2 = np.array(Image.open("test/data/RL_20210304_0.jpg"))
    print("hhh")

if __name__ == "__main__":
    
    plot = np.array(Image.open("res_plot.png"))
    image_grey = Image.open("res_render_grey.png")
    
    image_grey_np = np.array(image_grey)
    
    sought = [0,0,0]
    black  = np.count_nonzero(np.all(image_grey_np==sought,axis=2))
    print(f"black: {black}")
    
    sought = [255,255,255]
    white  = np.count_nonzero(np.all(image_grey_np==sought,axis=2))
    print(f"white: {white}")
    
    image_white = Image.open("res_render.png")
    
    pixels = [i for i in image_white.getdata()]
    assert not (0, 0, 0) in pixels, "(0, 0, 0) in pixels"
    
    image_white_bg = np.array(image_white)
    
    image_black_bg = copy.deepcopy(image_white_bg)
    
    image_black_bg[np.where((image_black_bg[...,0] == 255) & (image_black_bg[...,1] == 255) & (image_black_bg[...,2] == 255))] = [0,0,0]
    
    image_black = Image.fromarray(image_black_bg)
    
    pixels = [i for i in image_black.getdata()]
    assert not (255, 255, 255) in pixels, "(255, 255, 255) in pixels"
     
    frame = Image.open("image.png")
    
    frame_np = np.array(frame)
    
    extent = 0, plot.shape[1], plot.shape[0], 0
    
    plt.subplot(221)
    plt.imshow(frame_np, alpha=1, extent=extent, origin="upper")

    plt.subplot(222)
    im1 = plt.imshow(plot, extent=extent, origin="upper")
    im2 = plt.imshow(image_white_bg, alpha=0.5, extent=extent, origin="upper")
    
    plt.subplot(223)
    plt.imshow(image_white_bg, alpha=1, extent=extent, origin="upper")
    
    plt.subplot(224)
    plt.imshow(image_black_bg, alpha=1, extent=extent, origin="upper")
    plt.show()
    
    print('hhh')
    
    # invert_image = ImageOps.invert(image)
    # im = np.array(image)
    # invert_im = np.array(invert_image)
    
    # x, y, _ = np.where(invert_im)
    # min_x, max_x = min(x), max(x)
    # min_y, max_y = min(y), max(y)
    
    # crop = im[min_x:max_x, min_y:max_y, :]
    
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