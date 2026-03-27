# Prepare the data and finetune gpt-4o on openai

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.dataset_preparation import DatasetPreparation
import settings
from lib.utils import setup_log

skip_if_exist = True
skip_if_exist = False

def main():
    # Create output directory if it doesn't exist
    os.makedirs("../dataset", exist_ok=True)
    
    # Prepare the dataset
    dataset_prep = DatasetPreparation(settings)
    dataset_prep.run(skip_if_exist=skip_if_exist)
    

if __name__ == "__main__":
    setup_log()
    main()
