import cv2
import numpy as np
from video_capture import capture_video
from depth_estimation import estimate_depth
from midas.model_loader import load_model
from pointcloud import generate_point_cloud, visualize_point_cloud


# Load MiDaS model
model = load_model()

def main():
    for frame in capture_video():
        depth_map = estimate_depth(model, frame)
        depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Resize depth map to match frame size
        depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
        pcd = generate_point_cloud(frame, depth_map, scale=1.0)
        visualize_point_cloud(pcd)

        # Show original & depth map side by side
        combined = np.hstack((frame, depth_colored))
        cv2.imshow("Original | Depth Map", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()