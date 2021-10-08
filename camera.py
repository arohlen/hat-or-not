
# import the opencv library
import cv2
import keras
import numpy as np

model = keras.models.load_model('model.h5')
  
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, orig = vid.read()
  
    # Display the resulting orig


    img = cv2.resize(orig,(64,64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img,2)

    img = np.array([img])

    pred = model.predict(img)

    print(["No hat", "Hat"][np.argmax(pred)])

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (150,500)
    fontScale              = 3
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(orig,["No hat", "Hat"][np.argmax(pred)], 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    cv2.imshow('orig', orig)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()