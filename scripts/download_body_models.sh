#!/usr/bin/env bash

mkdir weights
mkdir data/mano -p
mkdir data/smplx -p
mkdir data/smpl -p
python scripts/download_data.py --target mano --out-folder data/mano && \
    unzip data/mano/mano_v1_2.zip -d data/mano && \
    mv data/mano/mano_v1_2/models/MANO_RIGHT.pkl data/mano && \
    rm -rf data/mano/mano_v1_2 data/mano/mano_v1_2.zip
python scripts/download_data.py --target smpl --out-folder data/smpl && \
    unzip data/smpl/mpips_smplify_public_v2.zip -d data/smpl && \
    mv data/smpl/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl data/smpl && \
    rm -rf data/smpl/mpips_smplify_public_v2.zip data/smpl/smplify_public
python scripts/download_data.py --target smplx --out-folder data/smplx && \
    unzip data/smplx/models_smplx_v1_1.zip -d data/smplx && \
    mv data/smplx/models/smplx/SMPLX_NEUTRAL.pkl data/smplx && \
    rm -rf data/smplx/models_smplx_v1_1.zip data/smplx/models
