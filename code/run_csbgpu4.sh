#!/bin/bash
#command_1. they had run on degree based negative sampled training data
#python main_archive.py --dddecoder 'dedicom' --encoder 'global'
#python main_archive.py --dddecoder 'bilinear' --encoder 'global'
#python main_archive.py --dddecoder 'bilinear' --encoder 'local'
#python main_archive.py --dddecoder 'dedicom' --encoder 'local'


#command_2 #here in config file only gnn algo was true
#python main_archive.py --sampling 'semi_random'  --dddecoder 'dedicom' --encoder 'local' --train --force-cvdir
#python main_archive.py --sampling 'degree_based'  --dddecoder 'dedicom' --encoder 'local' --train --force-cvdir



#command_3 #here in config file only deepsynergy algo is true
#python main_archive.py --config "/home/tasnina/Projects/SynVerse/code/config-files/master_config_csbgpu4.yaml" --sampling 'semi_random'   --train
#python main_archive.py --config "/home/tasnina/Projects/SynVerse/code/config-files/master_config_csbgpu4.yaml"  --sampling 'degree_based'  --train

#command_4 run gnn model with train-val-test split and on nndecoder
#07/13/2021 4:18 PM
#python main.py --config "/home/tasnina/Projects/SynVerse/code/config-files/master_config_csbgpu4.yaml" --sampling 'semi_random' --dddecoder 'nndecoder'  --train

python main.py --config "/home/tasnina/Projects/SynVerse/code/config-files/master_config_csbgpu4.yaml" --sampling 'semi_random' --train

