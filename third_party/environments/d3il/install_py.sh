set -euo pipefail

pip install -U pip setuptools wheel

pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.10.0 torchvision==0.25.0

pip install pybullet pyyaml scipy opencv-python matplotlib gin-config
pip install "pip<24" "setuptools<66" "wheel>=0.36.2,<0.38"
pip install gym==0.21.0

pip install --upgrade --ignore-installed blinker
pip install scikit-learn addict pandas plyfile tqdm open3d
pip install einops hydra-core==1.1.1 wandb termcolor ipython torchsde torchdiffeq
pip install imageio mujoco==2.3.2

pip install -e .