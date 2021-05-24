import io
import picamera
import cv2
import numpy

#สร้างความจำชื่อ stream เพื่อไม่ต้องมีไฟลหลายไฟลตอนจับหน้า
stream = io.BytesIO()

#ตั้งค่าความละเอียดตรงนี้
with picamera.PiCamera() as camera:
    camera.resolution = (320, 240)
    camera.capture(stream, format='jpeg')

#แปลงภาพให้เป็น numpy array
buff = numpy.frombuffer(stream.getvalue(), dtype=numpy.uint8)


image = cv2.imdecode(buff, 1)


#นำ cascade มาใช้เพื่อจับหน้า
face_cascade = cv2.CascadeClassifier('/home/pi/facedetection/haarcascade_frontalface_default.xml')

#Convert to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#ตรวจสอบหน้าจากการใช้ cascade
#หากไม่มีจะเป็น 0
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

print ("Found {}" + str(len(faces)) + " face(s)")

#วาดสี่เหลี่ยมรอบหน้า
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),4)

cv2.imwrite('result.jpg',image)