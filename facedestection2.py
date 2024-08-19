import cv2
import psycopg2
import hashlib
from mtcnn import MTCNN
import numpy as np
from datetime import datetime
import face_recognition
from sklearn.cluster import DBSCAN
from scipy.spatial import distance

# PostgreSQL database connection details
conn_params = {
    'dbname': 'PYTHONOPENCV',
    'user': 'postgres',
    'password': 'Varun@4545',
    'host': 'Subhash',
    'port': '5432'
}

def connect_to_db():
    """Connect to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**conn_params)
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Database connection error: {error}")
        return None

def create_image_hash(image):
    """Create a hash for the face image."""
    _, buffer = cv2.imencode('.jpg', image)
    image_hash = hashlib.md5(buffer).hexdigest()
    return image_hash

def insert_face_image(conn, face_image, image_hash, timestamp):
    """Insert face image into the database."""
    cursor = conn.cursor()
    cursor.execute("INSERT INTO faces (image, image_hash, timestamp) VALUES (%s, %s, %s)",
                   (face_image, image_hash, timestamp))
    conn.commit()
    cursor.close()

def retrieve_differentfaces(conn):
    """Retrieve face images and their timestamps from the differentfaces table."""
    cursor = conn.cursor()
    cursor.execute("SELECT image, merge_timestamp FROM differentfaces")
    images_with_timestamps = cursor.fetchall()
    cursor.close()
    return images_with_timestamps

def label_face(image, label):
    """Label the detected face with the given text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 255, 0)  # Green color
    thickness = 1

    # Ensure label is a string
    if not isinstance(label, str):
        label = str(label)

    # Determine text size
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size

    # Put text on the image
    cv2.putText(image, label, (10, 30), font, font_scale, color, thickness)

def compute_distance(encoding1, encoding2):
    """Compute Euclidean distance between two face encodings."""
    return distance.euclidean(encoding1, encoding2)

def main():
    """Main function to detect faces, compare them with database encodings, and label them."""
    # Initialize the MTCNN detector
    detector = MTCNN()

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    # Connect to the database
    conn = connect_to_db()
    if not conn:
        print("Failed to connect to the database")
        return

    # Retrieve face images and timestamps from the differentfaces table
    db_face_images = retrieve_differentfaces(conn)

    # Compute face encodings from the database images
    db_face_encodings = []
    db_timestamps = []
    for image_data, timestamp in db_face_images:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:
            db_face_encodings.append(encodings[0])
            db_timestamps.append(timestamp)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        # Convert the image to RGB (MTCNN expects RGB images)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = detector.detect_faces(img_rgb)

        # Collect face bounding boxes
        boxes = np.array([face['box'] for face in faces])

        # Apply clustering to reduce multiple detections
        if boxes.shape[0] > 0:
            # Use DBSCAN clustering
            clustering = DBSCAN(eps=20, min_samples=1).fit(boxes)
            labels = clustering.labels_
            unique_labels = set(labels)

            for label in unique_labels:
                # Get the bounding boxes for this cluster
                cluster_boxes = boxes[labels == label]
                # Combine boxes in the cluster (e.g., using average)
                x_min = int(np.min(cluster_boxes[:, 0]))
                y_min = int(np.min(cluster_boxes[:, 1]))
                x_max = int(np.max(cluster_boxes[:, 0] + cluster_boxes[:, 2]))
                y_max = int(np.max(cluster_boxes[:, 1] + cluster_boxes[:, 3]))

                # Extract face image from combined bounding box
                face_image = img[y_min:y_max, x_min:x_max]
                face_image_encoded = cv2.imencode('.jpg', face_image)[1].tobytes()
                face_image_hash = create_image_hash(face_image)
                timestamp = datetime.now().isoformat()

                # Convert face image to RGB and get face encoding
                face_img_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_encoding = face_recognition.face_encodings(face_img_rgb)

                # Initialize label as Unknown
                label = "Unknown"

                if face_encoding:
                    face_encoding = face_encoding[0]
                    # Compare face encoding with database encodings
                    for db_encoding, db_timestamp in zip(db_face_encodings, db_timestamps):
                        # Compute distance between encodings
                        dist = compute_distance(db_encoding, face_encoding)
                        print(f"Distance from DB encoding: {dist}")

                        # Check if the distance is within the acceptable range
                        # Adjust this threshold based on your needs
                        if dist < 0.6:  # Example threshold for 50% similarity
                            label = db_timestamp
                            break

                # Draw rectangle and label for the cluster
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label_face(img, label)

                # Insert face image into the database
                insert_face_image(conn, face_image_encoded, face_image_hash, timestamp)

        # Display the image with rectangles and labels
        cv2.imshow("image", img)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()
