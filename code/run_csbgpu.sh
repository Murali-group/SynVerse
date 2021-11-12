#!/bin/bash
##here in config file only the gnn should_run=True
#python main_archive.py --config "/home/tasnina/Projects/SynVerse/code/config-files/master_config_csbgpu3.yaml"  --dddecoder 'dedicom' --encoder 'local' --sampling 'semi_random'   --train
#
##06/28/2021: 3:27PM
#python main_archive.py --config "/home/tasnina/Projects/SynVerse/code/config-files/master_config_csbgpu3.yaml"  --dddecoder 'dedicom' --encoder 'local' --sampling 'semi_random'   --eval


#run gnn model with train-val-test split
#07/13/2021 4:50 PM
#python main.py --config "/home/tasnina/Projects/SynVerse/code/config-files/master_config_csbgpu3.yaml" --dddecoder 'dedicom' --sampling 'semi_random'  --train


#run deepsynergy on train-test-val-split of leave-combo type split
#08/16/2021 14:10
#python main.py --config "/home/grads/tasnina/Projects/SynVerse/code/config-files/master_config_csbgpu.yaml" --sampling 'semi_random' --train --use-genex --use-target

python main_with_test.py --config "/home/grads/tasnina/Projects/SynVerse/code/config-files/master_config_csbgpu.yaml" --sampling 'semi_random' --train
