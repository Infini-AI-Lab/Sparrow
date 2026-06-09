set -x 

# math related 
python polaris.py --local_dir $pwd/data/polaris_53k 
python test_aime2024.py --local_dir $pwd/data/aime2024 
python test_aime2025.py --local_dir $pwd/data/aime2025 
python test_aime2026.py --local_dir $pwd/data/aime2026 
python test_amc24.py --local_dir $pwd/data/amc24 
python test_amc23.py --local_dir $pwd/data/amc23 
python test_amc.py --local_dir $pwd/data/amc 
python test_gsm8k.py --local_dir $pwd/data/gsm8k 
python test_math500.py --local_dir $pwd/data/math500 

# code related 
python data_process.py --local_dir "$pwd/r1_livecodebench" --tasks livecodebench
python data_process.py --local_dir "$pwd/r1_humanevalplus" --tasks humanevalplus
python data_process.py --local_dir "$pwd/r1_mbppplus" --tasks mbppplus