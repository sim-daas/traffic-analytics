# ROS2 and NVIDIA DeepStream Integration for Traffic Analytics

This project provides a framework for building a traffic analytics dashboard using **NVIDIA DeepStream** for optimized inference and **ROS2** for modularity and communication. It's designed for easy addition of custom features.

*This setup is part of a larger integration effort.*

## **Prerequisites**
- NVIDIA GPU with CUDA support
- Docker installed with GPU support: [NVIDIA Docker Setup](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html)
- Ubuntu 20.04+ recommended
- Python 3.8+

---

## **Setup Instructions**

### **1. Get Docker Image**
Pull the prebuilt image with DeepStream and ROS2:
```bash
docker pull kamleshvkumarsingh/deepstream_ros2:latest
```

### **2. Launch Container**
Run with GPU access and display forwarding:
```bash
docker run --gpus all -it --rm --net=host --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -w /opt/nvidia/deepstream/deepstream/ \
    kamleshvkumarsingh/deepstream_ros2:latest
```

### **3. Install Dependencies (Inside Container)**
```bash
sudo apt update
# Install essential libraries and CUDA toolkit
sudo apt install -y python3-gi python3-dev python3-gst-1.0 python-gi-dev git meson python3 python3-pip python3.10-dev cmake g++ build-essential libglib2.0-dev libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev cuda-toolkit-12-3
# Install Python packages
pip3 install onnx onnxslim onnxruntime ultralytics matplotlib
```

---

## **YOLO Setup for DeepStream**

### **4. Prepare YOLOv8 Model**
```bash
# Clone necessary repos
git clone https://github.com/ultralytics/ultralytics.git
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git

# Copy export utility
cp DeepStream-Yolo/utils/export_yoloV8.py ultralytics/.

# Navigate to ultralytics directory
cd ultralytics

# Download weights (e.g., yolov8m)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt

# Export to ONNX
python3 export_yoloV8.py -w yolov8m.pt --dynamic --simplify

# Move ONNX model and labels back to DeepStream-Yolo directory (adjust paths as needed)
mv yolov8m.pt.onnx ../DeepStream-Yolo/
mv labels.txt ../DeepStream-Yolo/
cd ../DeepStream-Yolo
```

### **5. Compile YOLO Custom Plugins**
```bash
export CUDA_VER=12.3 # Set your CUDA version
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

### **6. Configure and Run**
1.  **Edit Config:** Update `config_infer_primary_yoloV8.txt` (or similar) in your DeepStream application directory with the correct paths to the generated `yolov8m.pt.onnx` model and `labels.txt`.
2.  **Run Example:** (Assuming you have a DeepStream Python app configured)
    ```bash
    # Example using a test application (adjust path and config file)
    cd /path/to/your/deepstream_python_app/
    python3 your_app.py your_video.mp4
    ```
    *Or using `deepstream-app`:*
    ```bash
    # Example using deepstream-app (adjust path and config file)
    cd /path/to/your/deepstream_config/
    deepstream-app -c your_deepstream_app_config.txt
    ```

---

## **Project Goal: Traffic Analytics Dashboard**

This setup forms the basis for a dashboard analyzing traffic patterns. DeepStream handles the heavy lifting of object detection/tracking, while ROS2 provides the communication backbone for processing results, adding custom logic (like counting vehicles, speed estimation), and potentially visualizing data.

---
