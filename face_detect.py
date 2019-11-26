import sys
import cv2


def main():
    # Usage: python face_detect.py shutterstock_faces.jpg haarcascade_frontalface_default.xml
    if len(sys.argv) <= 2:
        print("Usage: python face_detect.py shutterstock_faces.jpg haarcascade_frontalface_default.xml")
        return 0

    # pass in the image and cascade names as command-line arguments. 
    img_path = sys.argv[1]
    casc_path = sys.argv[2]

    # we create the cascade and initialize it with our face cascade. This loads the face cascade into memory so it’s ready for use. Remember, the cascade is just an XML file that contains the data to detect faces.
    face_cascade = cv2.CascadeClassifier(casc_path)

    # Here we read the image and convert it to grayscale. Many operations in OpenCV are done in grayscale.
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print("Found {} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # In the end, we display the image and wait for the user to press a key.
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()