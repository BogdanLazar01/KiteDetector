import base64
import tempfile

import streamlit as st
import cv2
import os
from PIL import Image
from ultralytics import YOLO

def process_video(input_path, output_path, delay):
    model = YOLO('./best.torchscript')
    threshold = 0.5

    video = cv2.VideoCapture(input_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    st.text("Running YoloV8 model on the video...")
    progress_bar = st.progress(0)

    for frame_count in range(total_frames):
        ret, frame = video.read()

        if not ret:
            break

        # Convert the frame to PIL Image
        image = Image.fromarray(frame)

        # Perform kite detection
        results = model(image)[0]

        # Draw bounding boxes on the frame
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Write frame to output video
        out.write(frame)

        # Show detections per frame in the Streamlit app
        st.text(f"Frame {frame_count + 1}/{total_frames}: {len(results)} detections")

        # Update progress bar
        progress_bar.progress((frame_count + 1) / total_frames)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    video.release()
    out.release()

    st.text("YoloV8 detection completed successfully.")
    pass

def main():
    st.title("Kite Detection App")

    # Upload video file
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        # Get the file paths
        video_path = temp_file.name
        output_path = "output.mp4"

        delay = 100

        # Use cv2.setHeadless() to enable headless mode
        cv2.setHeadless()
        
        # Process the video and generate output
        process_video(video_path, output_path, delay)

        # Provide download link for the output video file
        st.markdown(get_download_link(output_path, 'Download output video'), unsafe_allow_html=True)

        # Close and remove the temporary file
        temp_file.close()
        os.remove(temp_file.name)

    st.write("Upload a video file to get started!")


def get_download_link(file_path, text):
    with open(file_path, 'rb') as f:
        data = f.read()
    href = f'<a href="data:file/mp4;base64,{base64.b64encode(data).decode()}" download="{file_path}">{text}</a>'
    return href


if __name__ == "__main__":
    main()
