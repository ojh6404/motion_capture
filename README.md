# motion_capture

```sh
conda create -n mocap python=3.9 -y
conda activate mocap
conda install cuda cudnn -c conda-forge -y
# then install proper torch like below
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```


You need [MANO](https://mano.is.tue.mpg.de/), [SMPL](https://smpl.is.tue.mpg.de/), [SMPLX](https://smpl-x.is.tue.mpg.de/) model to run mocap.
Register and set env variables like below

```sh
export MANO_USERNAME="your@register.email.com"
export MANO_PASSWORD="password"
export SMPL_USERNAME="your@register.email.com"
export SMPL_PASSWORD="password"
export SMPLX_USERNAME="your@register.email.com"
export SMPLX_PASSWORD="password"
```

Then run prepare script
```sh
./prepare.sh
```

Now you can run mocap
```sh
python examples/hand_detection_example.py --input examples/test.jpg --mocap wilor --output wilor.jpg
```
