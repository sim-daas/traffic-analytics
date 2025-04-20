# Framework integrating ROS2 and NVIDIA DeepStream

This project integrates **ROS2**, **Gazebo**, and **NVIDIA DeepStream**. The setup supports advanced tracking and computer vision tasks using **DeepStream** within a ROS2-based control framework. 

## **Prerequisites**
- NVIDIA GPU with CUDA support
- Docker installed with GPU support - https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html
- Ubuntu 20.04+ recommended
- Python 3.8 or later

---

## **Setup Instructions**

### **1. Pull the Docker Image**
To get started, pull the prebuilt Docker image containing DeepStream, ROS2, and Gazebo:

```bash
docker pull kamleshvkumarsingh/deepstream_ros2:latest
```

### **2. Launch the Docker Container**
Run the container with GPU support and necessary permissions:

```bash
docker run --gpus all -it --rm --net=host --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -w /opt/nvidia/deepstream/deepstream/ \
    kamleshvkumarsingh/deepstream_ros2:latest
```

### **3. Run a DeepStream Application**
To run a DeepStream demo:

```bash
cd /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app
deepstream-app -c source30_1080p_dec_infer-resnet_tiled_display_int8.txt
```

For Python-based DeepStream applications:

```bash
cd /opt/nvidia/deepstream/deepstream/samples/deepstream_python_apps/apps/deepstream-test1
python3 deepstream_test_1.py ../../../streams/sample_720p.h264
```

---

## **Install Additional Dependencies**
Inside the container, run the following commands to set up additional dependencies:

### **4. Install CUDA Toolkit**
```bash
sudo nala install cuda-toolkit-12-3
```

### **5. Install Python Packages**
```bash
pip3 install onnx onnxslim onnxruntime ultralytics
pip install --upgrade matplotlib ultralytics
```

---

## **YOLO Setup for DeepStream**
### **6. Clone Required Repositories**
```bash
git clone https://github.com/ultralytics/ultralytics.git
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
```

### **7. Prepare YOLOv8 for DeepStream**
1. Copy the YOLOv8 export utility:
   ```bash
   cp DeepStream-Yolo/utils/export_yoloV8.py ultralytics/.
   ```

2. Download the YOLOv8 model weights:
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt
   ```

3. Export the YOLOv8 model to ONNX format:
   ```bash
   python3 export_yoloV8.py -w yolov8m.pt --dynamic --simplify
   ```

4. Move the generated ONNX model and label files:
   ```bash
   cp /home/ultralytics/yolov8m.pt.onnx .
   cp /home/ultralytics/labels.txt .
   ```

---

### **8. Compile YOLO Custom Plugins**
Set the CUDA version and compile the YOLO plugins:
```bash
export CUDA_VER=12.3
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

---

### **9. Configure and Run DeepStream with YOLO**
1. Edit the YOLOv8 inference configuration file:
   - Update `infer_yoloV8.txt` with paths to the ONNX model and labels.

2. Run the DeepStream application:
   ```bash
   git clone https://github.com/sim-daas/roswithai.git
   cd roswithai
   deepstream-app -c deepstream_app_config.txt
   ```

---

## **Features**
- **ROS2 Nodes**: Seamlessly integrate DeepStream into a ROS2 control framework.
- **Gazebo Simulation**: Simulate robot behavior and sensor data for testing.
- **NVIDIA DeepStream**: Real-time video analytics with YOLOv8-based inference.

---

## **Future Improvements**
- **Version Alignment**: Simplify the setup by unifying ROS2, Gazebo, and DeepStream versions.
- **Container Orchestration**: Use Kubernetes or Docker Compose to streamline deployments.
- **Enhanced Documentation**: Provide comprehensive guides for reproducibility and troubleshooting.

--- 

## **Feedback**
Your feedback and contributions are welcome! Please share your thoughts or submit issues to improve this project.
