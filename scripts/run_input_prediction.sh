python run_input.py --model $1 --category high
python run_input.py --model $1 --category medium
python run_input.py --model $1 --category low

python parse_results_input.py --model $1