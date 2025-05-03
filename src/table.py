#imports
import pandas as pd


class Table(object):
    def __init__(self, meta, predictions, actuals,num_weeks=20):
        self.meta = meta #(N,F), number of features, dataframe
        self.num_weeks = num_weeks #int
        self.predictions = predictions #(N,num_weeks), dict of numpy arrays
        self.actuals = actuals #(N,num_weeks), numpy array
        
    def by_planting_id(self, planting_id_list):
        idx = self.meta.index.get_indexer(planting_id_list)
        valid = idx != -1

        valid_idx = idx[valid]

        if len(valid_idx) == 0:
            print(f"No valid indices found for planting_id_list: {planting_id_list}, returning None")
            return None
        elif len(valid_idx) < len(planting_id_list):
            diff = len(planting_id_list) - len(valid_idx)
            print(f" {diff} indices not found for planting_id_list: {planting_id_list}, returning None")
            return None

        filtered = Table(self.meta.iloc[valid_idx], {key:data[valid_idx] for key,data in self.predictions.items()}, self.actuals[valid_idx],self.num_weeks)

        return filtered

    def by_type(self, type_list):
        labels = self.meta.index[self.meta['Type'].isin(type_list)]
        return self.by_planting_id(labels)

    def by_ranch(self, ranch_list):
        labels = self.meta.index[self.meta['Ranch'].isin(ranch_list)]
        return self.by_planting_id(labels)
    
    def by_variety(self, variety_list):
        labels = self.meta.index[self.meta['Variety'].isin(variety_list)]
        return self.by_planting_id(labels)

    