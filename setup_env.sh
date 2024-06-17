conda create -p ./rapids-24.04 -c rapidsai -c conda-forge -c nvidia \
  rapids=24.04 python=3.10 cuda-version=11.8 pytorch -y
conda activate -p ./rapids-24.04
pip install -r requirements.txt
