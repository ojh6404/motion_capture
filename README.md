# motion_capture

```sh
conda create -n mocap python=3.9 -y
conda activate mocap
conda install cuda cudnn -c conda-forge -y
# then install proper torch like below
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

./prepare.sh

python examples/hand_detection_example.py --input examples/test.jpg --mocap wilor --output wilor.jpg
```
