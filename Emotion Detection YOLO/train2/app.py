from ultralytics import YOLO
import cv2

def main():
    face_haar_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    model=YOLO("best.pt")
    camera = cv2.VideoCapture(0)
    while True: 
        r, t_img = camera.read()
        if not r:
            continue
        grey_image=cv2.cvtColor(t_img,cv2.COLOR_BGR2RGB)
        face_detected=face_haar_cascade.detectMultiScale(grey_image,1.32,5)
        for (x,y,w,h) in face_detected:
            cv2.rectangle(t_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            results = model(t_img)
            best_class_key=results[0].probs.top1
            names_dict=results[0].names
            class_index = names_dict[best_class_key]
            p_val=round(results[0].probs.top1conf.item(),2)

            cv2.putText(t_img,f"Class: {class_index}, Prob: {p_val}" , (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", t_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' key to exit
            break
    camera.release()
    cv2.destroyAllWindows()
       

if __name__ == "__main__":
    main()