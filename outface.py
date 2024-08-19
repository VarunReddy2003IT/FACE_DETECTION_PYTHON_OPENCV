import cv2
import psycopg2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

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
        print(error)
        return None

def retrieve_face_images(conn):
    """Retrieve face images and their timestamps from the faces table."""
    cursor = conn.cursor()
    cursor.execute("SELECT image, merge_timestamp FROM differentfaces")
    images = cursor.fetchall()
    cursor.close()
    return images

def resize_image(image, size=(100, 100)):
    """Resize image to a fixed size."""
    return cv2.resize(image, size)

def add_timestamp_label(image, timestamp, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.4, color=(0, 255, 0), thickness=1):
    """Add a timestamp label to an image."""
    text = f"{timestamp}"
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    text_x = 10
    text_y = text_height + 10
    image_with_text = cv2.putText(image.copy(), text, (text_x, text_y), font, font_scale, color, thickness)
    return image_with_text

def create_image_with_timestamp(images, image_size=(200, 200), timestamp_column_width=200):
    """Create an image with timestamp in a separate column."""
    rows = []

    for image_data, timestamp in images:
        # Convert byte data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize image
        img_resized = resize_image(img, size=image_size)

        # Create a blank image for the timestamp
        timestamp_image = np.zeros((image_size[1], timestamp_column_width, 3), dtype=np.uint8)

        # Add timestamp label to the timestamp image
        timestamp_image_with_text = add_timestamp_label(timestamp_image, timestamp)

        # Stack the image and timestamp side by side
        combined_row = np.hstack((img_resized, timestamp_image_with_text))
        rows.append(combined_row)

    # Create a vertical stack of rows
    if len(rows) > 0:
        combined_image = np.vstack(rows)
    else:
        combined_image = np.zeros((image_size[1], image_size[0] + timestamp_column_width, 5), dtype=np.uint8)

    return combined_image

def display_combined_image(combined_image):
    """Display the combined image in a scrollable window using Tkinter."""
    # Convert the image to RGB format
    combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

    # Convert the image to a PIL Image
    pil_image = Image.fromarray(combined_image_rgb)

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Images with Timestamps")

    # Create a canvas and scrollbars
    canvas = tk.Canvas(root)
    h_scrollbar = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
    v_scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)

    # Create a frame to hold the canvas
    frame = tk.Frame(canvas)
    frame.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=frame, anchor="nw")

    # Add the image to the frame
    tk_image = ImageTk.PhotoImage(pil_image)
    image_label = tk.Label(frame, image=tk_image)
    image_label.image = tk_image  # Keep a reference to prevent garbage collection
    image_label.pack()

    canvas.pack(side="left", fill="both", expand=True)
    h_scrollbar.pack(side="bottom", fill="x")
    v_scrollbar.pack(side="right", fill="y")

    canvas.config(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
    canvas.config(width=800, height=600)  # Set canvas size
    canvas.update_idletasks()  # Update the canvas size after packing

    # Start the Tkinter event loop
    root.mainloop()

def main():
    """Main function to connect to the database, retrieve images, and display them."""
    conn = connect_to_db()
    if conn:
        images = retrieve_face_images(conn)
        combined_image = create_image_with_timestamp(images)
        display_combined_image(combined_image)
        conn.close()
    else:
        print("Failed to connect to the database")


if __name__ == "__main__":
    main()
