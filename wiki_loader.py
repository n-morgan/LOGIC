import pandas as pd
import random

class WikiLoader:
    DATASET_PATH = '/data/sls/scratch/nmorgan/datasets/wiki_dpr_multiset/data/psgs_w100/multiset/train-00000-of-00157.parquet'

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def retrieve_samples(self):
        # Load the dataset
        df = pd.read_parquet(self.DATASET_PATH, engine='fastparquet')
        
        # Ensure num_samples does not exceed the dataset size
        num_samples = min(self.num_samples, len(df))
        
        # Return a random sample
        return df.sample(n=num_samples, random_state=random.randint(0, 10000))

# Example usage

if __name__ == "__main__":

    loader = WikiLoader(num_samples=5)
    samples = loader.retrieve_samples()
    print(samples)
