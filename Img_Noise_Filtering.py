import numpy as np
import cv2
import imutils
import random

# Generating grainy image
def blur_img(image_to_blur):
    height = image_to_blur.shape[0]
    width = image_to_blur.shape[1]
    for i in range(height):
        for j in range(width):
            if i % 4 == 0 and j % 4 == 0: # Every 4th column and row put some noise in there
                image_to_blur[i,j] = (random.randint(0,255),random.randint(0,255),random.randint(0,255)) # random noise

    return image_to_blur

def write_Name(image):
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    height = (image.shape[0])
    width = (image.shape[1])
    # Putting Last Name on top left corner
    cv2.putText(image, "Ghuman", (25, 25), font, 1, (150, 75, 240), 1)
    return image

# Median Filter Algorithm
def median_filter3by3(grainy_image):
    height,width,colors = grainy_image.shape # Getting the height,width,color channels of the blurred image
    img2 = np.zeros([height,width,colors],dtype="uint8") # Initializing the image to reconstruct
    # Pad the blurred image with values of 0 thickness of 1
    padded_img2 = cv2.copyMakeBorder(grainy_image,1,1,1,1,borderType=cv2.BORDER_CONSTANT,value=0)
    #v2.imshow("Padded Image",img1)

    for i in range(1,height+1): # We want to shift center of kernel and use that as starting index
        for j in range(1,width+1):
            for c in range(colors): # Looping through color channels
                # Manually defined 3 x 3 kernel with array indexing
                kernel = np.array([padded_img2[i+1,j-1,c],padded_img2[i+1,j,c],padded_img2[i+1,j+1,c],padded_img2[i,j-1,c],padded_img2[i,j,c],padded_img2[i,j+1,c],
                          padded_img2[i-1,j-1,c],padded_img2[i-1,j,c],padded_img2[i-1,j+1,c]]).flatten()
                img2[i-1,j-1,c] = np.median(kernel) # actual image needs to start from first pixel and find median

    img_new2 = img2.astype(np.uint8)
    return img_new2

def median_filter5by5(grainy_image_1):
    height,width,colors = grainy_image_1.shape # Getting the height,width, color channels of the blurred image
    img1 = np.zeros([height,width,colors],dtype="uint8") # Initializing the image to reconstruct
    # Pad the blurred image with values of 0 thickness of 2 for 5 x 5 kernel
    padded_img2 = cv2.copyMakeBorder(grainy_image_1,2,2,2,2,borderType=cv2.BORDER_CONSTANT,value=0)
    #cv2.imshow("Padded Image",img1)
    for i in range(2,height+2):
        for j in range(2,width+2):
            for c in range(colors):
                # Dynamically defining 5 x 5 kernel
                kernel = padded_img2[i-2:i+3,j-2:j+3,c].flatten()
                img1[i-2,j-2,c] = np.median(kernel) # Finding median - no need to sort

    img_new1 = img1.astype(np.uint8)
    return img_new1

def main():
    img = cv2.imread("IMG_2279.jpg")  # Original Image
    resized_img = imutils.resize(img, width=800)  # Resizing image to be smaller but maintaining aspect ratio
    image_text = resized_img.copy()
    image_blur = resized_img.copy()

    written_on_img = write_Name(image_text)
    grainy_img = blur_img(image_blur)
    clear_img_3by3 = median_filter3by3(grainy_img)
    clear_img_5by5 = median_filter5by5(grainy_img)

    cv2.imshow("Text Image", written_on_img)
    cv2.imwrite("Question#1_2.jpg", written_on_img)
    cv2.imshow("Grainy Image", grainy_img)
    cv2.imwrite("Question#1_3.jpg", grainy_img)
    cv2.imshow("Unblurred Image 3by3", clear_img_3by3)
    cv2.imwrite("Question#1_4_3by3.jpg",clear_img_3by3) # 3 by 3 is blurry but better than 5 x 5
    cv2.imshow("Unblurred Image 5by5",clear_img_5by5)
    cv2.imwrite("Question#1_4_5x5.jpg",clear_img_5by5) # 5 x 5 Kernel is worse than 3 by 3 m
    cv2.waitKey(0)

if __name__ == '__main__':
    main()