import cv2
import os

number_of_picture = 1
def process_video(video_path, output_folder):
    global number_of_picture
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)  # Indeksiranje frejmova

    # Kreiranje izlaznog foldera ako ne postoji
    os.makedirs(output_folder, exist_ok=True)

    # Analiza videa frejm po frejm
    while True:
        frame_num += 1

        grabbed, frame = cap.read()

        # Ako frejm nije zahvaćen
        if not grabbed:
            break

        # Ako je trenutni frejm svaki deseti frejm
        if frame_num % 10 == 0:
            # Generisanje naziva datoteke za čuvanje skrinšota
            filename = f"{output_folder}/pushup_{str(number_of_picture)}.png"

            # Čuvanje skrinšota
            cv2.imwrite(filename, frame)
            number_of_picture +=1

    cap.release()

if __name__ == '__main__':
    path = "data/Correct sequence/"
    for i in range(1, 51):
        video_path = path + "Copy of push up " + str(i) + ".mp4"
        output_path = "yolodemo/Images/PushUp"
        process_video(video_path, output_path)
