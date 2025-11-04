q1=ori
q2=ori
sample=1
temperature=0.0
eng_model=gpt-3.5-turbo


prompt_base=utils/prompt/standard/standard_base_aqua.txt
prompt_hint=utils/prompt/standard/standard_PHP_aqua.txt

python main.py --prompt_dir $prompt_base --eng $eng_model --seed 1357 --hint $prompt_hint --dataset aqua --num_test -1 --q1 $q1 --q2 $q2 --sample $sample --temp $temperature



prompt_base=utils/prompt/standard/standard_base_gsm8k.txt
prompt_hint=utils/prompt/standard/standard_PHP_gsm8k.txt

python main.py --prompt_dir $prompt_base --eng $eng_model --seed 1357 --hint $prompt_hint --dataset addsub --num_test -1 --q1 $q1 --q2 $q2 --sample $sample --temp $temperature

python main.py --prompt_dir $prompt_base --eng $eng_model --seed 1357 --hint $prompt_hint --dataset multiarith --num_test -1 --q1 $q1 --q2 $q2 --sample $sample --temp $temperature

python main.py --prompt_dir $prompt_base --eng $eng_model --seed 1357 --hint $prompt_hint --dataset singleeq --num_test -1 --q1 $q1 --q2 $q2 --sample $sample --temp $temperature

python main.py --prompt_dir $prompt_base --eng $eng_model --seed 1357 --hint $prompt_hint --dataset svamp --num_test -1 --q1 $q1 --q2 $q2 --sample $sample --temp $temperature

python main.py --prompt_dir $prompt_base --eng $eng_model --seed 1357 --hint $prompt_hint --dataset gsm8k --num_test -1 --q1 $q1 --q2 $q2 --sample $sample --temp $temperature

