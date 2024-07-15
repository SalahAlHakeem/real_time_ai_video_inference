

``` shell
# Intiating Yolov8: Make sure that you have `yolov8` folder with required content before running init command
cd yolov8/nvdsinfer_custom_impl_Yolo
sudo CUDA_VER=12.2 make clean
sudo CUDA_VER=12.2 make all
``` 


``` shell
# Compilation and running
sudo make
sudo ./main 3 configurations/primary_engine_config.yml  rtsp://192.168.1.121 rtsp://192.168.1.121
```


