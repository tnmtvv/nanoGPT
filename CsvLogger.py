import csv
import os

class CsvLogger:
    def __init__(self, filename='training_log.csv', bs=64, lr=0.02, opt_name='AdamW', seed=42):
        self.filename = filename
        self.bs = bs
        self.lr = lr
        self.opt = opt_name
        self.seed = seed
        
        header = ['iter_num', 'train_loss', 'val_loss', 'learning_rate', 
                  'model_flops_utilization', 'batch_size', 'optimizer_name', 'seed']

        # Write header if file doesn't exist
        directory = os.path.dirname(filename)
        
        # Create directory if it doesn't exist (only if directory is not empty string)
        if directory and not os.path.exists(directory):
            print('here')
            os.makedirs(directory, exist_ok=True)
        
        # Write header if file doesn't exist
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def report_scalar(self, title, series, value, iteration):
        # Initialize values dict in __init__ instead of using hasattr
        if not hasattr(self, 'values'):
            self.values = {}
        
        # Create nested dict structure properly
        if iteration not in self.values:
            self.values[iteration] = {}
        
        # Store the value with composite key
        key = f'{title}_{series}'
        self.values[iteration][key] = value
        
        # Define expected keys (ensure these match what's actually being reported)
        expected = ['train_loss', 'val_loss', 'learning_rate_lr', 'model_flops_utilization_mfu_percent']
        
        # Check if all expected values are present for this iteration
        if all(k in self.values[iteration] for k in expected):
            with open(self.filename, 'a', newline='') as f:
                # Extract values safely
                train_loss = self.values[iteration]['train_loss']
                val_loss = self.values[iteration]['val_loss']
                lr = self.values[iteration]['learning_rate_lr']
                mfu = self.values[iteration]['model_flops_utilization_mfu_percent']
                
                # Write the row
                f.write(f"{iteration},{train_loss},{val_loss},{lr},{mfu},{self.bs},{self.opt},{self.seed}\n") 
                # Clean up to save memory
                del self.values[iteration]
    