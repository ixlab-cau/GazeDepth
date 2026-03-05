# 1) Create a virtual environment
conda create -n groundedsam2 python=3.10 -y
conda activate groundedsam2

# 2) Install PyTorch (check the official PyTorch website for the command matching your CUDA version)
# Example:
pip install torch torchvision torchaudio

# 3) Clone the repository
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2

# 4) Download checkpoints (using the repository scripts)
cd checkpoints && bash download_ckpts.sh
cd ../gdino_checkpoints && bash download_ckpts.sh
cd ..

# 5) Install the packages
pip install -e .
pip install --no-build-isolation -e grounding_dino