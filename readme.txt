# Tool gán nhãn vật thể và thư viện training
pip install numpy==1.23.5
pip install PyQt5 sip
pip install lxml
pip install ultralytics opencv-python-headless
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117


#lệnh train
yolo train model=yolo11.pt data=data.yaml epochs=50 imgsz=640 device=0 amp=False #GPU=0