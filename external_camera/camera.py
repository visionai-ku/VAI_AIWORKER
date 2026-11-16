import pyrealsense2 as rs
import cv2
import numpy as np

def start_cameras(serials):
    pipelines = []
    for serial in serials:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        pipelines.append((serial, pipeline))
        print(f"Camera {serial} started")
    
    try:
        while True:
            for serial, pipeline in pipelines:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    print(f"No color frame from {serial}")
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                cv2.imshow(f"Camera {serial}", color_image) 
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for serial, pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    serials= ["409122271560"]
    start_cameras(serials)