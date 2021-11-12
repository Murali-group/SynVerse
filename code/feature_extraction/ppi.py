import argparse
import yaml
from collections import defaultdict
import os
import sys
import copy
import time
import pandas as pd
sys.path.insert(0,"/home/tasnina/Projects/SynVerse")
from sars2net.src.setup_datasets import setup_dataset_files
from sars2net.src.FastSinkSource.src import setup_sparse_networks
def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        config_map = yaml.load(conf)
    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to download and parse input files, and (TODO) run the FastSinkSource pipeline using them.")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="config-files/master-config.yaml",
                       help="Configuration file for this script.")
    group.add_argument('--download-only', action='store_true', default=False,
                       help="Stop once files are downloaded and mapped to UniProt IDs.")
    group.add_argument('--force-download', action='store_true', default=False,
                       help="Force re-downloading and parsing of the input files")
    return parser
    
    
def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    dataset_settings = config_map['dataset_settings']
    datasets_dir = dataset_settings['datasets_dir']
    
    string_settings = config_map['string_network_preparation_settings']
    string_dir = string_settings['string_dir']
    string_file_path = string_dir+string_settings['string_net_files']
    cutoff= string_settings['string_cutoff']
    net_type = string_settings['string_nets'] #combined

    # Download and parse the ID mapping files 
    # Download, parse, and map (to uniprot) the network files 
    setup_dataset_files(datasets_dir, dataset_settings['datasets_to_download'], dataset_settings.get('mappings'), **kwargs)
    
    sparse_networks, net_names, nodes = setup_sparse_networks.setup_sparse_networks([], string_net_files=[string_file_path], string_nets=[net_type], string_cutoff=cutoff)
    
    print(sparse_networks[0:5])
    
    
    sparse_nets_file = string_dir + 'c'+str(cutoff)+'_'+net_type+'_sparse_net.mat' 
    net_names_file = string_dir + 'c'+str(cutoff)+'_'+net_type+'_net_names.txt'
    node_ids_file = string_dir + 'c'+str(cutoff)+'_'+net_type+'_node_ids.txt'
    
    setup_sparse_networks.write_sparse_net_file( sparse_networks, sparse_nets_file, net_names, net_names_file, nodes, node_ids_file)
if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)