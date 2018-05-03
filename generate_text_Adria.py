from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import random
import glob

"""
    Generates synthetic text applying some distortions.
    Params:
        text: string containing the text to be drawn
        font: path to the font file
        font_size: size of the font
        text_color: color of the text
        background_color: color of the background
        color_variance: the maximum color variance (applies to both text and background)
            (all color values are grayscale, from 0 to 255)

    You can add more parameters for gaussian noise, gamma correction and affine transform if needed.

"""
def generate_text(text, font, font_size, text_color=50, background_color=180, color_variance=60):

    font = ImageFont.truetype(font, font_size)

    # Gets font metrics to determine width and height.
    # It's broken for some fonts due to PIL bugs.
    # To make sure there's enough space, width and height are multiplied by 3
    # then the image is cropped.
    # Also, there are some fonts which are rendered incompletely, even given enough space.
    # It's a bit crappy, that's why we wanted to move to Pango
    # In any case, it should work MOST of the time
    (_, _), (offset_x, offset_y) = font.font.getsize(text)
    pil_width, pil_height = font.getsize(text)

    TW = pil_width-offset_x
    TH = pil_height-offset_y

    # generate background
    imo = Image.new('L',(3*TW,3*TH),color=int(background_color+(np.random.uniform(color_variance) - (color_variance/2))))

    # background sin noise Tro add
    #imo = Image.new('L', (3*TW, 3*TH), color=255)

    # generate clean image to find out its bounding box
    imtmp = Image.new('L',(3*TW,3*TH),color=255)
    draw = ImageDraw.Draw(imtmp)
    draw.text((TW, TH), text, font=font, fill=0)
    imtmp = np.array(imtmp)
    ret,thresh = cv2.threshold(imtmp,254,255,cv2.THRESH_BINARY_INV)

    # generate noisy image
    draw = ImageDraw.Draw(imo)
    draw.text((TW, TH), text, font=font, fill=int(text_color+(np.random.uniform(color_variance) - (color_variance/2))))
    imo = np.array(imo)
    h,w = imo.shape
    im = cv2.GaussianBlur(imo,(3,3),1)

    # add gaussian noise
    mean = 0
    var = 255
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(h,w))
    noisy1 = np.uint8(np.clip(np.float32(im) + gauss,0,255))

    # add random gamma correction
    gamma = np.random.uniform(.3,2) # randomizes gamma value
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    noisy = cv2.LUT(np.uint8(noisy1), table)

    #add random affine transform
    alpha=np.sqrt(TH*TH / 3.0)
    step=.02 # the higher the more aggressive the transformation is
    A=np.array([[TW + .5*TW,TH],
        [TW + (.5*TW)-alpha,TH + TH],
        [TW + (.5*TW)+alpha,TH + TH]],np.float32)
    B=np.array([[(TW + .5*TW)*np.random.uniform(1-step,1+step),(TH)*np.random.uniform(1-step,1+step)],
        [(TW + (.5*TW)-alpha)*np.random.uniform(1-step,1+step),(TH + TH)*np.random.uniform(1-step,1+step)],
        [(TW + (.5*TW)+alpha)*np.random.uniform(1-step,1+step),(TH + TH)*np.random.uniform(1-step,1+step)]],np.float32)
    M=cv2.getAffineTransform(A,B)
    warped=cv2.warpAffine(noisy,M,(w,h))

    # warp clean image to find out its bounding box
    warpedthresh = cv2.warpAffine(thresh,M,(w,h))
    points = np.argwhere(warpedthresh!=0)
    points = np.fliplr(points)
    r = cv2.boundingRect(np.array([points]))

    final_image = np.uint8(warped[r[1]:r[1]+r[3],r[0]:r[0]+r[2]])

    return final_image


# example usage
if __name__ == "__main__":
    baseDir = '/home/lkang/datasets/synthetic_Pau/fonts/'
    #img = generate_text('woe', baseDir+'Prof. Jorge.ttf', 75)
    #print(img.shape)

    cv2.namedWindow('display', 0)
    ttfs = glob.glob(baseDir+'*.ttf')
    while True:
        ttf = random.choice(ttfs)
        #ttf = '/home/lkang/datasets/synthetic_Pau/fonts/Doctor.ttf'
        img = generate_text('atong 3.', ttf, 75)
        print(img.shape)
        cv2.imshow('display', img)
        key = cv2.waitKey(0)
        if key & 0xff == ord(' '):
            continue
        elif key & 0xff == ord('q'):
            break
