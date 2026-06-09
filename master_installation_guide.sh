# before installation, you need to make sure your python version is 3.12, smaller than that usually hit installation errors 

# first install sglang 
cd sglang 
bash -x install.sh 

# second install vortex 
cd ../vortex 
pip install -e . --no-build-isolation --no-deps 

# third install verl 
cd .. 
pip install -e . 
pip install flash-attn --no-build-isolation 
