import cv2
import matplotlib.pyplot as plt
import numpy as np

def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(data_path, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (ML_Models_pred.txt), the format is the same as Sample.txt.
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    #raise NotImplementedError("To be implemented")
    f = open(data_path)
    gif = cv2.VideoCapture("data\detect\\video.gif")
    pf = open("ML_Models_pred.txt", "a")
    first_time = 1

    while True:
      
      ret, frame = gif.read()
      if not ret:
        break

      predictions = ""
      f.seek(0,0)
      row = f.readline()
      for i in range(0, int(row) - 1):
        temp = f.readline()
        temp = temp.strip("\n").split(" ")
        x1, y1, x2, y2, x3, y3, x4, y4 = temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7]
        img = crop(x1, y1, x2, y2, x3, y3, x4, y4, frame)
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (36, 16))
        if (clf.classify(img.flatten().reshape(1, -1)) == 1):
          if(first_time):
            pts = np.array([[int(x1), int(y1)], [int(x2), int(y2)], [int(x4), int(y4)], [int(x3), int(y3)]])
            cv2.polylines(frame, [pts], True, (0, 255, 0), thickness=2)
          predictions += "1 "
        else:
          predictions += "0 "
      
      predictions = predictions.strip(" ")
      predictions += "\n"
      #print(predictions)
      pf.write(predictions)
      if (first_time):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(frame)
        plt.show()
        first_time = 0
    
    pf.close()
    gif.release()
    cv2.destroyAllWindows()
    # End your code (Part 4)
