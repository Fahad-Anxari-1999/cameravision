import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import streamlit as st
import tempfile
import os
import csv
import time
import sys; print(sys.executable)

# Load YOLOv8 face detection model
@st.cache_resource
def load_face_model():
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    return YOLO(model_path)

# Load general YOLOv8 model for person detection
@st.cache_resource
def load_yolo_model():
    return YOLO('yolo11s.pt')

# Load facial expression recognition model
@st.cache_resource
def load_expression_model():
    processor = AutoImageProcessor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
    model = AutoModelForImageClassification.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
    return processor, model

face_model = load_face_model()
yolo_model = load_yolo_model()
expression_processor, expression_model = load_expression_model()

# Initialize the tracker (ByteTrack)
tracker = sv.ByteTrack()

# Initialize Supervision's PolygonZones for the areas of interest
queue_polygon = np.array([[52, 232], [76, 305], [246, 208], [335, 267], [121, 431], [149, 494], [175, 574], [305, 574], [463, 424], [421, 289], [262, 154]])
queue_zone = sv.PolygonZone(polygon=queue_polygon, triggering_anchors=[sv.Position.CENTER_RIGHT])
queue_annotator = sv.PolygonZoneAnnotator(zone=queue_zone, color=sv.Color(255, 0, 0))

shelf_polygon = np.array([[478, 319], [509, 432], [604, 333], [647, 190], [587, 156]])
shelf_zone = sv.PolygonZone(polygon=shelf_polygon, triggering_anchors=[sv.Position.CENTER_LEFT])
shelf_annotator = sv.PolygonZoneAnnotator(zone=shelf_zone, color=sv.Color(255, 0, 0))

shelf_polygon1 = np.array([[367, 209], [461, 319], [555, 212], [511, 134], [444, 138]])
shelf_zone1 = sv.PolygonZone(polygon=shelf_polygon1, triggering_anchors=[sv.Position.CENTER_LEFT])
shelf_annotator1 = sv.PolygonZoneAnnotator(zone=shelf_zone1, color=sv.Color(255, 0, 0))

shelf_polygon2 = np.array([[53, 132], [106, 230], [226, 148], [231, 90], [201, 62], [130, 83]])
shelf_zone2 = sv.PolygonZone(polygon=shelf_polygon2, triggering_anchors=[sv.Position.CENTER_LEFT])
shelf_annotator2 = sv.PolygonZoneAnnotator(zone=shelf_zone2, color=sv.Color(255, 0, 0))

shelf_polygon3 = np.array([[263, 115], [329, 213], [388, 142], [451, 126], [441, 70], [338, 86], [267, 87]])
shelf_zone3 = sv.PolygonZone(polygon=shelf_polygon3, triggering_anchors=[sv.Position.CENTER_LEFT])
shelf_annotator3 = sv.PolygonZoneAnnotator(zone=shelf_zone3, color=sv.Color(255, 0, 0))

counter_polygon = np.array([[74, 309], [82, 339], [117, 435], [306, 307], [324, 265], [245, 205], [76, 302]])
counter_zone = sv.PolygonZone(polygon=counter_polygon)
counter_annotator = sv.PolygonZoneAnnotator(zone=counter_zone, color=sv.Color(0, 0, 255))

# Initialize LineZone for tracking people crossing the line
start, end = sv.Point(x=51, y=176), sv.Point(x=82, y=270)
line_zone = sv.LineZone(start=start, end=end, triggering_anchors = [sv.Position.TOP_LEFT, sv.Position.TOP_RIGHT, sv.Position.BOTTOM_LEFT, sv.Position.BOTTOM_RIGHT])
line_annotator = sv.LineZoneAnnotator()

# Initialize LabelAnnotator for facial expressions
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

total_people_entered = 0
customer=0 
visitors=0
tracked_Customers=[]
tracked_Visitors=[]
current_Customers=0
current_Visitors=0
shelf_count1=0
shelf_count2=0
shelf_count3=0

def process_frame(frame, dwell_times, start_time, zone_entry_counts):
    global total_people_entered
    global customer
    global tracked_Customers
    global tracked_Visitors
    global visitors
    global current_Customers
    global current_Visitors
    global shelf_count1
    global shelf_count
    global shelf_count2
    global shelf_count3
    

    # Run YOLOv8 on the frame for general person detection
    results = yolo_model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Update detections with tracking information
    tracked_detections = tracker.update_with_detections(detections)

    # Filter for person detections (class_id == 0)
    person_detections = tracked_detections[tracked_detections.class_id == 0]

    # Trigger line zone and update entry and exit counts
    line_zone.trigger(person_detections)
    enter_count = line_zone.in_count
    exit_count = line_zone.out_count

    # Queue Monitoring
    is_in_queue_zone = queue_zone.trigger(person_detections)
    queue_count = int(np.sum(is_in_queue_zone))
    current_Customers=queue_count

    # Shelf Monitoring (Checking the number of persons in the shelf areas)
    is_in_shelf_zone = shelf_zone.trigger(person_detections)
    shelf_count = int(np.sum(is_in_shelf_zone))

    # Shelf Zone 1
    is_in_shelf_zone1 = shelf_zone1.trigger(person_detections)
    shelf_count1 = int(np.sum(is_in_shelf_zone1))

    # Shelf Zone 2
    is_in_shelf_zone2 = shelf_zone2.trigger(person_detections)
    shelf_count2 = int(np.sum(is_in_shelf_zone2))

    # Shelf Zone 3
    is_in_shelf_zone3 = shelf_zone3.trigger(person_detections)
    shelf_count3 = int(np.sum(is_in_shelf_zone3))
    current_Visitors= shelf_count+shelf_count1+shelf_count2+shelf_count3
    
    # Track and annotate dwell time only when in specific zones
    for i, (bbox, tracker_id) in enumerate(zip(person_detections.xyxy, person_detections.tracker_id)):
        if is_in_queue_zone[i] or is_in_shelf_zone[i] or is_in_shelf_zone1[i] or is_in_shelf_zone2[i] or is_in_shelf_zone3[i]:
            if tracker_id not in dwell_times:
                dwell_times[tracker_id] = 0
                start_time[tracker_id] = cv2.getTickCount()
                total_people_entered += 1

            # If person is in queue or shelf zone, update dwell time
            end_tick = cv2.getTickCount()
            elapsed_time = (end_tick - start_time[tracker_id]) / cv2.getTickFrequency()
            dwell_times[tracker_id] = elapsed_time

            # Annotate dwell time on the person's bounding box
            label_text = f"Time:{dwell_times[tracker_id]:.1f}s"
            annotated_frame = label_annotator.annotate(
                scene=frame,
                detections=person_detections[i:i + 1],
                labels=[label_text]
            )
            if is_in_shelf_zone[i] or is_in_shelf_zone1[i] or is_in_shelf_zone2[i] or is_in_shelf_zone3[i]:
                if tracker_id not in tracked_Visitors:
                    tracked_Visitors.append(tracker_id)
                    visitors+=1
            if is_in_queue_zone[i]:
                if tracker_id not in tracked_Customers:
                    tracked_Customers.append(tracker_id)
                    customer+=1
                    
            #while track in tracked_Customers:
               # if track in tracked_Visitors:
                   # visitors-=1
            
            
            

                

    # Annotate zones and counts
    annotated_frame = queue_annotator.annotate(scene=frame)
    annotated_frame = shelf_annotator.annotate(scene=annotated_frame)
    annotated_frame = shelf_annotator1.annotate(scene=annotated_frame)
    annotated_frame = shelf_annotator2.annotate(scene=annotated_frame)
    annotated_frame = shelf_annotator3.annotate(scene=annotated_frame)

    # Display statistics on the frame
    rect_x, rect_y, rect_w, rect_h = frame.shape[1] - 320, frame.shape[0] - 320, 220, 250
    cv2.rectangle(annotated_frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (255, 255, 255), -1)

    # Display text information for zone statistics
    cv2.putText(annotated_frame, f'Total People Zones: {total_people_entered}', (rect_x + 10, rect_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(annotated_frame, f'Customers: {customer}', (rect_x + 10, rect_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(annotated_frame, f'Visitors: {visitors}', (rect_x + 10, rect_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(annotated_frame, f'Queue Count: {queue_count}', (rect_x + 10, rect_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(annotated_frame, f'Shelf Count 1: {shelf_count}', (rect_x + 10, rect_y + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(annotated_frame, f'Shelf Count 2: {shelf_count1}', (rect_x + 10, rect_y + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(annotated_frame, f'Shelf Count 3: {shelf_count2}', (rect_x + 10, rect_y + 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(annotated_frame, f'Shelf Count 4: {shelf_count3}', (rect_x + 10, rect_y + 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(annotated_frame, f'Current Customers: {current_Customers}', (rect_x + 10, rect_y + 265), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(annotated_frame, f'Current Visitors: {current_Visitors}', (rect_x + 10, rect_y + 295), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return annotated_frame, queue_count, shelf_count


def process_video_stream(video_file):
    csv_file = 'analytics_data.csv'
    # Write CSV headers
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Total People Entered', 'Total Customers', 'Total Visitors', 'Queue Count', 'Shelf Count', 'Current Customers', 'Current Visitor', 'Shelf 1', 'Shelf 2', 'Shelf 3', "Shelf 4"])
    start_time_csv = time.time()
    dwell_times = {}
    start_time = {}
    zone_entry_counts = {}

    # Open video feed or file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        st.error("Error: Unable to open video file.")
        return

    # Retrieve video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object to save the processed video
    output_file = "full_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Create a Streamlit image placeholder
    image_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame, queue_count, shelf_count = process_frame(frame, dwell_times, start_time, zone_entry_counts)

        # Record data every 10 seconds
        current_time = time.time()
        if current_time - start_time_csv >= 10:
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)),
                    total_people_entered,
                    customer,
                    visitors,
                    queue_count,
                    shelf_count,
                    current_Customers,
                    current_Visitors,
                    shelf_count,
                    shelf_count1,
                    shelf_count2,
                    shelf_count3

                ])
            start_time_csv = current_time


        # Write the processed frame to the output video
        out.write(processed_frame)

        # Convert the frame from BGR to RGB (Streamlit expects RGB)
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Update the image in the Streamlit app
        image_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)

    cap.release()
    out.release()

    st.write(f"Processed video saved as {output_file}")

def main():
    st.title("YOLOv8 + Supervision Video Analysis")
    st.write("Upload a video for facial detection and expression analysis using YOLOv8 and ViT.")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        if st.button('Process Video'):
            with st.spinner('Processing video...'):
                process_video_stream(tfile.name)
            st.success('Video processing complete!')

        # Clean up the temporary file
        os.unlink(tfile.name)
    if st.button("Analysis Rush Hours"):
        data= pd.read_csv('analytics_data.csv')
        data['Total_Customers_Visitors'] = data['Current Customers'] + data['Current Visitor']
        max_customers_visitors = data['Total_Customers_Visitors'].max()
        max_period = data[data['Total_Customers_Visitors'] == max_customers_visitors]['Timestamp'].iloc[0]
        st.title(f'The Maximum Customers at Rush Hours : {max_customers_visitors}' )
        st.title(f'Rush Time : {max_period}' )
        shelf_columns = ['Shelf 1', 'Shelf 2', 'Shelf 3', 'Shelf 4']

        # Sum the clients for each shelf over the entire period
        shelf_totals = data[shelf_columns].sum()

        # Find the shelf with the maximum total clients
        max_shelf = shelf_totals.idxmax()
        max_count = shelf_totals[max_shelf] 
        st.title(f'{max_shelf} has mazimum of {max_count} customers ')




if __name__ == "__main__":
    main()
