#!/usr/bin/bash

git submodule update --init --recursive
pip install ninja
pip install -e .
pip install -U "setuptools<70"

# install hand object detector
echo "Installing hand object detector"
cd third_party/hand_object_detector && pip install -r requirements.txt
rm -rf lib/pycocotools && rm -rf lib/datasets # remove pycocotools and datasets cause they may cause conflicts with existing pycocotools
gdown https://drive.google.com/uc\?id\=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE
cd lib && python setup.py build develop --user && cd ..

# install hamer
echo "Installing HaMeR"
cd ../hamer && pip install -e .[all]
bash fetch_demo_data.sh
mkdir _DATA/data/mano -p
gdown https://drive.google.com/uc\?id\=1sgZ9dF0FH5z9wSXm9dNXuSyN3ZaTX28U -O _DATA/data/mano/MANO_RIGHT.pkl -c
# remove unnecessary files
rm -rf hamer_demo_data.tar.gz
rm -rf _DATA/hamer_demo_data.tar.gz
rm -rf _DATA/vitpose_ckpts

# install wilor
echo "Installing WiLoR"
cd ../WiLoR && pip install -e .
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
gdown https://drive.google.com/uc\?id\=1sgZ9dF0FH5z9wSXm9dNXuSyN3ZaTX28U -O mano_data/MANO_RIGHT.pkl -c


# install 4D-Humans
echo "Installing 4D-Humans"
cd ../4D-Humans && pip install -e .[all]
mkdir data && gdown https://drive.google.com/uc\?id\=1LBRm4pZzB7gp5aSPr_M-kTI2mrKMi483 -O data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl -c
pip install -U PyOpenGL PyOpenGL_accelerate git+https://github.com/ojh6404/chumpy.git@patch-python3.11
