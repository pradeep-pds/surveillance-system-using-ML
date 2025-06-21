import cv2
import csv
import os
import time
import datetime
import pandas as pd
from tkinter import *
from tkinter import messagebox as mess
from tkinter import ttk

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_haarcascadefile():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess._show(title='Missing File', message='haarcascade_frontalface_default.xml not found!')
        exit()
def load_motion_entries():
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    filename = f"MotionData/MotionData_{date}.csv"

    # Clear the treeview rows
    for row in tv.get_children():
        tv.delete(row)

    if os.path.isfile(filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            for line in reader:
                if len(line) >= 7:
                    tv.insert('', 'end', values=(line[2], line[4], line[6]))
def TrackImages():
    check_haarcascadefile()
    assure_path_exists("MotionData/")
    assure_path_exists("FaceDetails/")

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.isfile("TrainingImageLabel/Trainner.yml"):
        mess._show(title='Data Missing', message='Please click on Save Profile to reset data!!')
        return
    recognizer.read("TrainingImageLabel/Trainner.yml")

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    if not os.path.isfile("FaceDetails/FaceDetails.csv"):
        mess._show(title='Details Missing', message='People details are missing, please check!')
        return

    df = pd.read_csv("FaceDetails/FaceDetails.csv")

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    capture = None

    while True:
        ret, im = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID = str(ID)[1:-1]
                bb = str(aa)[2:-2]
                capture = [str(ID), '', bb, '', date, '', timeStamp]
                cv2.putText(im, bb, (x, y+h), font, 1, (255, 255, 255), 2)
            else:
                bb = "Unknown"
                cv2.putText(im, bb, (x, y+h), font, 1, (0, 0, 255), 2)

        cv2.imshow('Video Surveillance', im)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    # Save capture info to CSV if recognized
    if capture:
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        filename = f"MotionData/MotionData_{date}.csv"
        header = ['Id', '', 'Name', '', 'Date', '', 'Time']

        file_exists = os.path.isfile(filename)
        with open(filename, 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(capture)

        load_motion_entries()
def main():
    global tv

    root = Tk()
    root.title("Motion Detection System")
    root.geometry("700x500")

    columns = ('Name', 'Date', 'Time')
    tv = ttk.Treeview(root, columns=columns, show='headings')
    for col in columns:
        tv.heading(col, text=col)
        tv.column(col, width=200)
    tv.pack(fill=BOTH, expand=True)

    btn_start = Button(root, text="Start Motion Detection", command=TrackImages)
    btn_start.pack(pady=20)

    load_motion_entries()

    root.mainloop()

if __name__ == '__main__':
    main()
