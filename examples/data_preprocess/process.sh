set -x 
python polaris.py --local_dir $pwd/data/polaris_53k 
python test_aime2025x4.py --local_dir $pwd/data/aime2025x4 
python test_aime2026x4.py --local_dir $pwd/data/aime2026x4 
python test_aime2024x4.py --local_dir $pwd/data/aime2024x4 
python test_amc24.py --local_dir $pwd/data/amc24 
python test_amc23.py --local_dir $pwd/data/amc23 
python test_amc.py --local_dir $pwd/data/amc 
python test_gsm8k.py --local_dir $pwd/data/gsm8k 
python test_math500.py --local_dir $pwd/data/math500 
