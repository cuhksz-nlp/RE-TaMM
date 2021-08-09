#train
python re_tamm_main.py --do_train --do_eval --task_name sample --data_dir ./data/sample_data/ --model_path ./bert_model_path --dep_order second_order --model_name RE_TaMM.SAMPLE.BERT.L --do_lower_case

# test
python re_tamm_main.py --do_test --task_name sample --data_dir ./data/sample_data/ --model_path ./RE_TaMM.SAMPLE.BERT.L/