 #!/usr/bin/bash

git submodule update --init --recursive
pip install ninja
pip install -e .
pip install "setuptools<70" # for hand object detector build

# install body models
echo "Installing body models"
./scripts/download_body_models.sh

# install hand object detector
echo "Installing hand object detector"
cd third_party/hand_object_detector && pip install -r requirements.txt
rm -rf lib/pycocotools && rm -rf lib/datasets # remove pycocotools and datasets cause they may cause conflicts with existing pycocotools
cd lib && python setup.py build develop && cd ../../..
gdown https://drive.google.com/uc\?id\=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE -O weights/hand_object_detector.pth # download hand object detector checkpoints

# install hamer
echo "Installing HaMeR"
pip install -e "third_party/hamer[all]"
gdown https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT # download hamer checkpoints and data
tar --warning=no-unknown-keyword --exclude=".*" -xvf hamer_demo_data.tar.gz
mv _DATA/data/mano_mean_params.npz data/mano/
mv _DATA/hamer_ckpts/checkpoints/hamer.ckpt weights/hamer.ckpt
rm -rf hamer_demo_data.tar.gz _DATA

# install wilor
echo "Installing WiLoR"
pip install -e "third_party/WiLoR"
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -O weights/wilor.ckpt # download wilor checkpoints

# install hamba
echo "Installing Hamba"
pip install -e "third_party/Hamba[all]" "causal-conv1d>=1.4.0"
pip install "git+https://github.com/state-spaces/mamba"
pip install -e "third_party/VMamba/kernels/selective_scan"
gdown https://drive.google.com/uc\?id\=1JRPC11YfQym8t_EZkhsroglvGHrGPbU- -O hamba.zip
unzip hamba.zip && mv hamba/checkpoints/hamba.ckpt weights/hamba.ckpt && rm -rf hamba hamba.zip

# install 4D-Humans
echo "Installing 4D-Humans"
pip install -e "third_party/4D-Humans[all]"
wget https://people.eecs.berkeley.edu/~jathushan/projects/4dhumans/hmr2_data.tar.gz && mkdir -p data/4D-Humans && \
    tar --warning=no-unknown-keyword --exclude=".*" -xvf hmr2_data.tar.gz -C data/4D-Humans && \
    mv data/4D-Humans/data/smpl_mean_params.npz data/smpl && \
    mv data/4D-Humans/data/SMPL_to_J19.pkl data/smpl && \
    mv 'data/4D-Humans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt' weights/hmr2.ckpt && \
    rm -rf hmr2_data.tar.gz data/4D-Humans

# install smplx
echo "Installing SMPL-X"
pip install -e "third_party/smplx"

# patch chumpy and renderer
pip install -U PyOpenGL PyOpenGL_accelerate git+https://github.com/ojh6404/chumpy.git@patch-python3.11 git+https://github.com/facebookresearch/pytorch3d.git
