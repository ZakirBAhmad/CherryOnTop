#imports
import pandas as pd
import numpy as np



class CherryTable(object):
    """
    desc: A class to handle and manipulate tabular data for cherry predictions and actuals
    params:
        meta: DataFrame of shape (N,F) containing feature metadata
            meta data contains PlantingID as the index, Ranch, Class, Type, Variety, Year, TransplantWeek
        predictions: Dictionary of numpy arrays of shape (N,num_weeks) containing predictions
            the weeks are weeks after transplant
        actuals: Numpy array of shape (N,num_weeks) containing actual values
            the weeks are weeks after transplant
        num_weeks: Integer specifying number of weeks to track (default 20)
    returns: None
    """
    def __init__(self, meta, predictions, actuals,num_weeks=20):
        self.meta = meta #(N,F), number of features, dataframe
        self.num_weeks = num_weeks #int
        self.predictions = predictions #(N,num_weeks), dict of numpy arrays
        self.actuals = actuals #(N,num_weeks), numpy array
        
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
    
    def group_by(self, ranch=False, class_=False, type_=False, variety=False):
        """
        desc: Groups and aggregates the table data by specified categories. If all parameters are False,
              returns a new table with all data summed into a single row.
        params:
            ranch: Boolean indicating whether to group by ranch
            class_: Boolean indicating whether to group by class
            type_: Boolean indicating whether to group by type
            variety: Boolean indicating whether to group by variety
        returns:
            CherryTable: New table with data grouped and summed by specified categories
        """
        # Determine grouping columns based on flags
        group_cols = []
        if ranch:
            group_cols.append('Ranch')
        if class_:
            group_cols.append('Class') 
        if type_:
            group_cols.append('Type')
        if variety:
            group_cols.append('Variety')
            
        if not group_cols:
            # If no grouping columns, sum the entire dataset
            summed_meta = pd.DataFrame(self.meta.sum(numeric_only=True)).T
            summed_preds = {key: pred.sum(axis=0, keepdims=True) for key, pred in self.predictions.items()}
            summed_actuals = self.actuals.sum(axis=0, keepdims=True)
            return CherryTable(
                summed_meta,
                summed_preds,
                summed_actuals,
                self.num_weeks
            )
            
        # Group meta data
        grouped_meta = self.meta.groupby(group_cols).sum(numeric_only=True)
        
        # Group predictions
        grouped_preds = {}
        for key, pred in self.predictions.items():
            grouped_pred = np.zeros((len(grouped_meta), self.num_weeks))
            for i, (idx, group) in enumerate(self.meta.groupby(group_cols).groups.items()):
                grouped_pred[i] = pred[group].sum(axis=0)
            grouped_preds[key] = grouped_pred
            
        # Group actuals
        grouped_actuals = np.zeros((len(grouped_meta), self.num_weeks))
        for i, (idx, group) in enumerate(self.meta.groupby(group_cols).groups.items()):
            grouped_actuals[i] = self.actuals[group].sum(axis=0)
            
        return CherryTable(
            grouped_meta,
            grouped_preds,
            grouped_actuals,
            self.num_weeks
        )