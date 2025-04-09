import pandas as pd
import os

class DatasetUtils:
    
    @staticmethod
    def get_dataframe_distributed(dataset_base_dir):
        category_number = 0
        dataframe = pd.DataFrame(columns = ["sample_path", "label"])
        for label in os.listdir(dataset_base_dir):
            category_number += 1
            sample_dir = os.path.join(dataset_base_dir, label)
            for sample_name in os.listdir(sample_dir):
                sample_path = os.path.join(sample_dir, sample_name)
                new_row = pd.DataFrame({
                    "sample_path": [sample_path],
                    "label": [label]
                })
                
                dataframe = pd.concat([dataframe,new_row],
                                      ignore_index=True)
        
        return dataframe, category_number