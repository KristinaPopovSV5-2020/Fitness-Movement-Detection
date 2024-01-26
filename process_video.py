import cv2
import os

number_of_picture = 142
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
            filename = f"{output_folder}/crunch_{str(number_of_picture)}.png"

            # Čuvanje skrinšota
            cv2.imwrite(filename, frame)
            number_of_picture +=1

    cap.release()

if __name__ == '__main__':
    path = "data/"
    for i in range(14, 16):
        video_path = path + "crunch_" + str(i) + ".mp4"
        output_path = "yolodemo/Images/Crunch"
        process_video(video_path, output_path)
    print(number_of_picture)
