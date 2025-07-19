import cv2
import os
import time
import smtplib
import firebase_admin
import pandas as pd
import numpy as np
from firebase_admin import credentials, firestore
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from twilio.rest import Client

def start_motion_detection_with_recognition():
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")

  
    df = pd.read_csv("FaceDetails/FaceDetails.csv")

    
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    
    if not firebase_admin._apps:
        cred = credentials.Certificate("survillance-monitoring-firebase-adminsdk-fbsvc-cfd53f5177.json")
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    def send_email(subject, body, recipient_email, attachment=None):
        sender_email = "user-email"
        sender_password = ""  # App-specific password

        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = recipient_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        if attachment:
            with open(attachment, 'rb') as img_file:
                message.attach(MIMEImage(img_file.read(), name=os.path.basename(attachment)))

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
            print("Email sent.")
            server.quit()
        except Exception as e:
            print("Email failed:", str(e))

    def send_sms(sms_body, recipient_phone_number):
        account_sid = ""
        auth_token = ""
        sender_phone_number = ""
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=sms_body, from_=sender_phone_number, to=recipient_phone_number
        )
        print("SMS sent:", message.sid)

    def send_data_to_firebase(collection_name, data):
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        doc_ref = db.collection(collection_name).document(f"time{current_time}")
        doc_ref.set(data)
        print("Data sent to Firebase.")

    # Start camera
    video = cv2.VideoCapture(0)
    first_frame = None

    while True:
        check, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None:
            first_frame = blur
            continue

        delta = cv2.absdiff(first_frame, blur)
        thresh = cv2.threshold(delta, 50, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 775:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            motion_detected = True

        if motion_detected:
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                serial, conf = recognizer.predict(face_img)

                if conf < 50:  # Known face
                    name = df.loc[df['SERIAL NO.'] == serial]['NAME'].values[0]
                    print(f"Recognized: {name}")
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:  # Unknown face
                    print("Unknown face detected!")
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"Suspicious/suspicious_{timestamp}.jpg"
                    if not os.path.exists("Suspicious"):
                        os.makedirs("Suspicious")
                    cv2.imwrite(filename, frame)
                    body = f"Unknown person detected at {timestamp}"
                    send_email("Unknown Face Alert", body, "user-email", attachment=filename)
                    send_sms(body, "mobile-number")
                    send_data_to_firebase("Motion_Time", {"status": "unknown", "time": timestamp})

        cv2.imshow("Monitoring", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
