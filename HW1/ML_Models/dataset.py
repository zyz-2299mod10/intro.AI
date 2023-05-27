import os
import cv2

def load_images(data_path):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    #raise NotImplementedError("To be implemented")
    dataset = []
    for filename in os.listdir(data_path):
        if(filename == "car"):
            dir = os.path.join(data_path, filename)
            for img_name in os.listdir(dir):
                img_dir = os.path.join(dir, img_name)
                img = cv2.imread(img_dir)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                img_gray_resize = cv2.resize(img_gray, (36, 16))
                
                dataset.append((img_gray_resize, 1))
        if(filename == "non-car"):
            dir = os.path.join(data_path, filename)
            for img_name in os.listdir(dir):
                img_dir = os.path.join(dir, img_name)
                img = cv2.imread(img_dir)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                img_gray_resize = cv2.resize(img_gray, (36, 16))
                
                dataset.append((img_gray_resize, 0))
    # End your code (Part 1)
    return dataset
