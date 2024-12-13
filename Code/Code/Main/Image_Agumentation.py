import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import read
import matplotlib.pyplot as plt

def rotation(img):

    Original_Image = Image.open(img)

    # The Image
    rotated_image = Original_Image.transpose(Image.ROTATE_90)


    return rotated_image


def resizing(img):

    Img=cv2.imread(img)

    stretch_near = cv2.resize(Img, (780, 540),
                              interpolation=cv2.INTER_LINEAR)

    return stretch_near

def translation(img):
    image = cv2.imread(img)

    # Store height and width of the image
    height, width = image.shape[:2]

    quarter_height, quarter_width = height / 4, width / 4

    T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])

    # We use warpAffine to transform
    # the image using the matrix, T
    img_translation = cv2.warpAffine(image, T, (width, height))


    return img_translation


def random_erasing(Img):

    # read the input image
    img = Image.open(Img)

    # define a transform to perform three transformations:
    # convert PIL image to tensor
    # randomly select a rectangle region in a torch Tensor image
    # and erase its pixels
    # convert the tensor to PIL image
    transform = T.Compose(
        [T.ToTensor(), T.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
         T.ToPILImage()])
    # apply the transform on image
    img = transform(img)

    return img



def image_Aug(img):
    IMG=read.image(img)

    for i in range(len(IMG)):
        print("i :",i)

        rotated_img=rotation(IMG[i])

        rotated_img.save("Processed/Img_Aug/Rotation/"+str(i)+'.jpg')



        re_img=resizing(IMG[i])
        cv2.imwrite("Processed/Img_Aug/Resizing/" + str(i) + '.jpg', re_img)


        Trans_img=translation(IMG[i])
        cv2.imwrite("Processed/Img_Aug/Translation/" + str(i) + '.jpg', Trans_img)


        RE_img=random_erasing(IMG[i])
        RE_img.save("Processed/Img_Aug/Random_Erasing/" + str(i) + '.jpg')



def Images(img):

    #image_Aug(img)

    Rotation="Processed/Img_Aug/Rotation/*"

    Resizing="Processed/Img_Aug/Resizing/*"

    Translation="Processed/Img_Aug/Translation/*"

    Random_Erasing="Processed/Img_Aug/Random_Erasing/*"


    return Rotation,Resizing,Translation,Random_Erasing







