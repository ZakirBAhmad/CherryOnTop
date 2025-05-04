#imports
import pandas as pd
import numpy as np



class CherryTable(object):
    """
    desc: A class to handle and manipulate tabular data for cherry predictions and actuals
    params:
        meta: DataFrame of shape (N,F) containing feature metadata
            meta data contains PlantingID as the index, Ranch, Class, Type, Variety, Year, WeekTransplanted, Ha
        predictions: Dictionary of numpy arrays of shape (N,num_weeks) containing predictions
            the weeks are weeks after transplant
        actuals: Numpy array of shape (N,num_weeks) containing actual values
            the weeks are weeks after transplant
        num_weeks: Integer specifying number of weeks to track (default 20). If None, the table is seasonal
    returns: None
    """
    def __init__(self, meta, predictions, actuals = None,num_weeks=20):
        self.meta = meta #(N,F), number of features, dataframe
        self.num_weeks = num_weeks #int
        self.predictions = predictions #(N,num_weeks), dict of numpy arrays
        if actuals is not None:
            self.actuals = actuals #(N,num_weeks), numpy array
        else:
            self.actuals = np.zeros((len(self.meta), self.num_weeks))
        
    def by_planting_id(self, planting_id_list):
        """
        desc: Filters the table data by a list of planting IDs
        params:
            planting_id_list: List of planting IDs to filter by
        returns: 
            CherryTable: Filtered table containing only specified planting IDs, or None if invalid IDs
        """
        idx = self.meta.index.get_indexer(planting_id_list)
        valid = (idx != -1)

        valid_idx = idx[valid]

        if len(valid_idx) == 0:
            print(f"No valid indices found for planting_id_list: {planting_id_list}, returning None")
            return None
        
        elif len(valid_idx) < len(planting_id_list):
            diff = len(planting_id_list) - len(valid_idx)
            print(f" {diff} indices not found for planting_id_list: {planting_id_list}, returning None")
            return None

        filtered = CherryTable(
            self.meta.iloc[valid_idx],
            {key:data[valid_idx] for key,data in self.predictions.items()}, 
            self.actuals[valid_idx],
            self.num_weeks
            )

        return filtered

    def filter(self, ranches=None, classes=None, types=None, varieties=None):
        """
        desc: Filters the table data by ranch, class, type and variety
        params:
            ranches: List of ranch names to filter by
            classes: List of classes to filter by
            types: List of types to filter by
            varieties: List of varieties to filter by
        returns:
            CherryTable: Filtered table containing only rows matching the specified criteria
        """
        mask = pd.Series(True, index=self.meta.index)
        
        if ranches is not None:
            mask &= self.meta['Ranch'].isin(ranches)
        if classes is not None:
            mask &= self.meta['Class'].isin(classes)
        if types is not None:
            mask &= self.meta['Type'].isin(types)
        if varieties is not None:
            mask &= self.meta['Variety'].isin(varieties)
            
        labels = self.meta.index[mask]
        return self.by_planting_id(labels)
    
    def season_table(self):
        """
        Converts predictions and actuals from weeks-after-transplant to absolute weeks of the year.
        Shifts each row's values by its WeekTransplanted value, padding with zeros at the start.
        
        returns:
            CherryTable: New table with predictions and actuals aligned to calendar weeks
        """
        # Calculate max week needed (max transplant week + 20 prediction weeks)
        max_week = self.meta['WeekTransplanted'].max() + self.num_weeks
        
        # Initialize arrays for the full year
        season_preds = {}
        for key, pred in self.predictions.items():
            # Create indices for each row's target positions
            row_indices = np.arange(len(self.meta))[:, None]
            col_indices = np.arange(self.num_weeks)[None, :] + self.meta['WeekTransplanted'].values[:, None]
            
            # Initialize array and assign values using advanced indexing
            season_pred = np.zeros((len(self.meta), max_week))
            season_pred[row_indices, col_indices] = pred
            season_preds[key] = season_pred
            
        # Same approach for actuals
        season_actuals = np.zeros((len(self.meta), max_week))
        season_actuals[row_indices, col_indices] = self.actuals
        return CherryTable(
            self.meta,
            season_preds, 
            season_actuals,
            None
        )
        