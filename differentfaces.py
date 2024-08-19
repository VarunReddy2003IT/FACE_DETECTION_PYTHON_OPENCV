import cv2
import numpy as np
import hashlib
import psycopg2
import face_recognition
from datetime import datetime

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
        print(f"Error connecting to the database: {error}")
        return None

def retrieve_face_images(conn):
    """Retrieve face images and their timestamps from the faces table."""
    cursor = conn.cursor()
    cursor.execute("SELECT image, timestamp FROM faces")
    images_with_timestamps = cursor.fetchall()
    cursor.close()
    return images_with_timestamps

def create_image_hash(image):
    """Create a hash for the face image."""
    _, buffer = cv2.imencode('.jpg', image)
    image_hash = hashlib.md5(buffer).hexdigest()
    return image_hash

def face_exists_in_differentfaces(conn, image_hash):
    """Check if the face image hash already exists in the differentfaces table."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM differentfaces WHERE image_hash = %s", (image_hash,))
    count = cursor.fetchone()[0]
    cursor.close()
    return count > 0

def update_merged_face_image(conn, image_hash, timestamps):
    """Update the merged face image timestamps in the differentfaces table."""
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE differentfaces SET merge_timestamp = %s WHERE image_hash = %s",
        (timestamps, image_hash)
    )
    conn.commit()
    cursor.close()

def insert_merged_face_image(conn, face_image, image_hash, timestamps):
    """Insert the merged face image into the differentfaces table with timestamps."""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO differentfaces (image, image_hash, merge_timestamp) VALUES (%s, %s, %s)",
        (face_image, image_hash, timestamps)
    )
    conn.commit()
    cursor.close()

def truncate_faces_table(conn):
    """Truncate the faces table."""
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE faces")
    conn.commit()
    cursor.close()

def process_faces(conn, images_with_timestamps):
    """Process and cluster face images based on similarity and merge timestamps."""
    face_encodings = []
    face_images = []
    image_timestamps = []

    for image_data, timestamp in images_with_timestamps:
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize image for faster processing
            small_img_rgb = cv2.resize(img_rgb, (0, 0), fx=0.5, fy=0.5)

            # Get face encodings
            encodings = face_recognition.face_encodings(small_img_rgb)
            if encodings:
                face_encodings.extend(encodings)
                face_images.extend([img] * len(encodings))
                image_timestamps.extend([timestamp] * len(encodings))
        except Exception as e:
            print(f"Error processing image: {e}")
            continue

    # Perform clustering
    clusters = []
    visited = [False] * len(face_encodings)
    for i, encoding in enumerate(face_encodings):
        if visited[i]:
            continue
        cluster = [i]
        for j, other_encoding in enumerate(face_encodings):
            if i != j and not visited[j]:
                match = face_recognition.compare_faces([encoding], other_encoding, tolerance=0.6)
                if match[0]:
                    cluster.append(j)
                    visited[j] = True
        clusters.append(cluster)

    clustered_images = []
    for cluster in clusters:
        if cluster:
            # Combine images in the cluster
            first_image = face_images[cluster[0]]
            merged_image = cv2.resize(first_image, (640, 480))  # Example resize for simplicity

            # Collect and sort timestamps for the cluster
            cluster_timestamps = [image_timestamps[idx] for idx in cluster]
            sorted_timestamps = sorted(cluster_timestamps, reverse=True)  # Descending order

            # Convert list of timestamps to a string for storage
            sorted_timestamps_str = [timestamp.strftime('%Y-%m-%d %H:%M:%S') for timestamp in sorted_timestamps]
            timestamps_str = ', '.join(sorted_timestamps_str)

            # Create a hash for the merged image
            image_hash = create_image_hash(merged_image)

            # Check if the face image hash already exists in the database
            if not face_exists_in_differentfaces(conn, image_hash):
                clustered_images.append((merged_image, image_hash, timestamps_str))
            else:
                update_merged_face_image(conn, image_hash, timestamps_str)

    return clustered_images

def main():
    """Main function to process and store merged face images."""
    conn = connect_to_db()
    if conn:
        images_with_timestamps = retrieve_face_images(conn)
        clusters = process_faces(conn, images_with_timestamps)

        for image, image_hash, timestamps in clusters:
            merged_image_encoded = cv2.imencode('.jpg', image)[1].tobytes()

            # Insert new merged face image into the database
            insert_merged_face_image(conn, merged_image_encoded, image_hash, timestamps)

        # Truncate the faces table after processing
        truncate_faces_table(conn)

        conn.close()
    else:
        print("Failed to connect to the database")

if __name__ == "__main__":
    main()
