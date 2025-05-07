# run_all.py
# main script to run the entire small business survival prediction pipeline

import os
import sys
import subprocess
import time

def run_script(script_name, description):
    """run a python script and handle any errors"""
    print(f"\n{'='*80}")
    print(f"running {description}...")
    print(f"{'='*80}")
    
    start_time = time.time()
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f" {description} completed successfully in {elapsed:.2f} seconds")
            return True
        else:
            print(f" {description} failed with return code {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f" {description} failed with error: {e}")
        return False
    except Exception as e:
        print(f" error running {description}: {e}")
        return False

def create_directories():
    """create necessary directories if they don't already exist"""
    directories = ["../data", "../processed"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """run the entire pipeline"""
    print("small business survival prediction pipeline")
    print("-------------------------------------------")
    
    create_directories()
    
    # step 1: data processing # plspls workahhahehieuhaihfiuehf.
    if run_script("src/combine_bds_data.py", "data preparation"):
        # step 2: additional data processing
        run_script("src/logistic_script_data.py", "data aggregation")
        
        # step 3: preprocessing
        if run_script("src/preprocessing.py", "data preprocessing"):
            # step 4: model training and evaluation
            run_script("src/decision_tree.py", "decision tree model")
            run_script("src/linear_regression.py", "linear regression model")
            #run_script("src/logistic_regression.py", "logistic regression model")
            
            # step 5: model comparison
            #run_script("src/evaluation.py", "model evaluation and comparison")
            
            print("\n pipeline completed successfully!")
            print("\nresults are available in the ../outputs.plots directory:")
        else:
            print("\n pipeline could not complete due to preprocessing errors")
    else:
        print("\n pipeline could not complete due to data preparation errors")

if __name__ == "__main__":
    main()
