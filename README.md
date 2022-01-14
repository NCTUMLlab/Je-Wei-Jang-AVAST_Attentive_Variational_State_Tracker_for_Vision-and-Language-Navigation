# AVAST: Attentive VAriational State Tracker for Vision-and-Language Navigation

This is the PyTorch implementation for AVAST: Attentive VAriational State Tracker for Vision-and-Language Navigation from National Chiao Tung University, Taiwan.

---

## Installation

Clone the AVAST_R2R repository:
```bash
git clone --recursive https://github.com/weber12599/AVAST_R2R.git
cd AVAST_R2R
```

### Downloading pre-trained embedding and R2R datasets
```bash
cd Matterport3DSimulator
bash download.sh
```

### Downloading Matterport3D Dataset (optional)
To use original Matterport3D simulator you must first download the [Matterport3D Dataset](https://niessner.github.io/Matterport/) which is available after requesting access [here](https://niessner.github.io/Matterport/). The download script that will be provided allows for downloading of selected data types. At minimum you must download the `matterport_skybox_images`. If you wish to use depth outputs then also download `undistorted_depth_images` and `undistorted_camera_parameters`.

Set an environment variable to the location of the **unzipped** dataset, where <PATH> is the full absolute path (not a relative path or symlink) to the directory containing the individual matterport scan directories (17DRP5sb8fy, 2t7WUuJeko7, etc):

```bash
export MATTERPORT_DATA_DIR=<PATH>
```

### Building using Docker
```bash
docker build --rm -t r2r/avast:base .
docker run --name r2r -it --gpus all \
    -e DISPLAY -e="QT_X11_NO_MITSHM=1" -v /tmp/.X11-unix:/tmp/.X11-unix \
    --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/AVAST_R2R/data/v1/scans \
    --volume `pwd`:/root/mount/AVAST_R2R \
    --restart=unless-stopped --shm-size 32G \
    -p <ssh_port>:22 -p <tensorboard_port>:6006 r2r/avast:base
```

### Buiding Matterport3D simulator
Now (from inside the docker container), build the simulator code:
```bash
cd /root/mount/AVAST_R2R
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make
cd ../
```

### Buiding Lookup Table for Location Connectivity
Now (from inside the docker container), build the simulator code:
```bash
python3 tasks/data/scripts/generate_adj_dict.py > ./tasks/env/adj_dict/total_adj_dict.json
```

---

## Training and evaluation

### Pre-training state trackers
* Pre-training attentive state tracker:
    ```
    python3 ast_pre_train.py --mode pre_train --state_tracker ast --agent seq2seq
    ```
* pre-training attentive variational state tracker:
    ```
    python3 avast_pre_train.py --mode pre_train --state_tracker avast --agent seq2seq
    ```
    
### Fine-tuning an agent with RL algorithms
* Fine-tuning an agent with AST+REINFORCE:
    ```
    python3 reinforce_fine_tune.py --mode train --state_tracker ast --load_pre_trained_dir <ast_path> --agent reinforce
    ```   
* Fine-tuning an agent with AVAST+SACD+RECED:
    ```
    python3 sacd_fine_tune.py --mode train --state_tracker avast --load_pre_trained_dir <avast_path> --agent sacd --demo_activate --curriculum
    ```

### Evaluating AST+Seq2Seq
* Evaluating AST+Seq2Seq:
    ```
    python3 ast_pre_train.py --mode test --state_tracker ast --agent seq2seq --load_dir <model_path>
    ```    
* Evaluating AVAST+Seq2Seq:
    ```
    python3 avast_pre_train.py --mode test --state_tracker avast --agent seq2seq --load_dir <model_path>
    ```
* Evaluating AST+REINFORCE:
    ```
    python3 reinforce_fine_tune.py --mode test --state_tracker ast --agent reinforce --load_dir <model_path>
    ```
* Evaluating AVAST+SACD:
    ```
    python3 sacd_fine_tune.py --mode test --state_tracker avast --agent sacd --load_dir <model_path>
    ```