import pandas as pd
import imagehash
from PIL import Image
import os

class ReverseImageSearch:
    def __init__(self, img_path,db_path='db.csv'):
        self.img_path = img_path
        self.db_path = db_path
        self.db= self.load_database()
    def load_database(self): 
        return pd.read_csv(self.db_path)

    def compute_hashes(self):
        hashes={}
        hashes['phash']=int(str(imagehash.phash(Image.open(self.img_path))),16)
        hashes['dhash']= int(str(imagehash.dhash(Image.open(self.img_path))),16)
        hashes['whash']=int(str(imagehash.whash(Image.open(self.img_path))),16)
        return hashes
    def get_closest_match(self):
                
        phash_input,dhash_input,whash_input=self.compute_hashes().values()
        self.db['phash_dist'] = self.db['phash'].apply(lambda x: bin(x ^ phash_input).count("1"))
        self.db['dhash_dist'] = self.db['dhash'].apply(lambda x: bin(x ^ dhash_input).count("1"))
        self.db['whash_dist'] = self.db['whash'].apply(lambda x: bin(x ^ whash_input).count("1"))

        self.db['total_dist'] = self.db[['phash_dist', 'dhash_dist', 'whash_dist']].mean(axis=1)

        closest_match = self.db.loc[self.db['total_dist'].idxmin()]
        return closest_match


class ClothesRepository:
    def __init__(self, img_path, db_path='db.csv'):
        self.img_path = img_path
        self.db_path = db_path
        self.db = self.load_database()

    def load_database(self):
        if not os.path.exists(self.db_path):
            pd.DataFrame(columns=['img_path', 'price', 'phash', 'dhash', 'whash']).to_csv(self.db_path, index=False)
        return pd.read_csv(self.db_path)

    def insert_record(self, price):
        try:
            hashes = self.compute_hashes()
        except Exception as e:
            print(f"Failed to compute hashes: {e}")
            return

        new_record = {
            'img_path': self.img_path,
            'price': price,
            'phash': hashes['phash'],
            'dhash': hashes['dhash'],
            'whash': hashes['whash']
        }

        existing_index = self.db[self.db['img_path'] == self.img_path].index

        if not existing_index.empty:
            self.db.loc[existing_index[0]] = new_record  
        else:
            self.db = pd.concat([self.db, pd.DataFrame([new_record])], ignore_index=True)

        try:
            self.db[['phash', 'dhash', 'whash']] = self.db[['phash', 'dhash', 'whash']].astype('uint64')
        except Exception as e:
            print(f"Hash casting issue: {e}")

        self.db.to_csv(self.db_path, index=False)

    def compute_hashes(self):
        try:
            img = Image.open(self.img_path)
        except Exception as e:
            raise RuntimeError(f"Could not open image at {self.img_path}: {e}")

        return {
            'phash': int(str(imagehash.phash(img)), 16),
            'dhash': int(str(imagehash.dhash(img)), 16),
            'whash': int(str(imagehash.whash(img)), 16)
        }
