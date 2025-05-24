#imports
import pandas as pd
import numpy as np
from typing import Optional, List, Dict



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
    def __init__(self, meta: pd.DataFrame, predictions: Dict[str, np.ndarray], actuals: Optional[np.ndarray] = None, num_weeks: int = 20):
        self.meta = meta #(N,F), number of features, dataframe
        self.num_weeks = num_weeks #int
        self.predictions = predictions #(N,num_weeks), dict of numpy arrays
        if actuals is not None:
            self.actuals = actuals #(N,num_weeks), numpy array
        else:
            self.actuals = np.zeros((len(self.meta), self.num_weeks))
        
    def by_planting_id(self, planting_id_list: List[str]) -> Optional['CherryTable']:
        """
        desc: Filters the table data by a list of planting IDs
        params:
            planting_id_list: List of planting IDs to filter by
        returns:
            CherryTable: Filtered table containing only specified planting IDs, or None if invalid IDs
        """
        #getting indices
        idx = self.meta.index.get_indexer(planting_id_list)
        #valid indices
        valid = (idx != -1)
        #filtering valid indices
        valid_idx = idx[valid]
        #no valid indices
        if len(valid_idx) == 0:
            print(f"No valid indices found for planting_id_list: {planting_id_list}, returning None")
            return None
        #some indices not found
        elif len(valid_idx) < len(planting_id_list):
            diff = len(planting_id_list) - len(valid_idx)
            print(f" {diff} indices not found for planting_id_list: {planting_id_list}, returning None")
            return None
        #creating filtered table    
        filtered = CherryTable(
            self.meta.iloc[valid_idx],
            {key:data[valid_idx] for key,data in self.predictions.items()}, 
            self.actuals[valid_idx],
            self.num_weeks
            )

        return filtered

    def filter(self, ranch_list: Optional[List[str]] = None, class_list: Optional[List[str]] = None, 
              type_list: Optional[List[str]] = None, variety_list: Optional[List[str]] = None) -> Optional['CherryTable']:
        """
        desc: Filters the table data by ranch, class, type and variety
        params:
            ranch_list: List of ranch names to filter by
            class_list: List of classes to filter by
            type_list: List of types to filter by
            variety_list: List of varieties to filter by
        returns:
            CherryTable: Filtered table containing only rows matching the specified criteria
        """
        #creating mask
        mask = pd.Series(True, index=self.meta.index)
        #filtering by ranch
        if ranch_list is not None:
            mask &= self.meta['Ranch'].isin(ranch_list)
        #filtering by class
        if class_list is not None:
            mask &= self.meta['Class'].isin(class_list)
        #filtering by type
        if type_list is not None:
            mask &= self.meta['Type'].isin(type_list)
        #filtering by variety
        if variety_list is not None:
            mask &= self.meta['Variety'].isin(variety_list)
        #creating labels
        labels = self.meta.index[mask]
        #filtering by labels
        return self.by_planting_id(list(labels))
    
    def season_table(self):
        """
        desc: Converts predictions and actuals from weeks-after-transplant to absolute weeks of the year.
              Shifts each row's values by its WeekTransplanted value, padding with zeros at the start.
        params: None
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
            self.num_weeks
        )
        
    def summary(self,ranches=False, classes=False, types=False, varieties=False,include_actuals=True):
        """
        desc: Creates a summary DataFrame containing total predictions and actuals for each planting,
              along with per-hectare yield calculations. Can be grouped by ranch, class, type and variety.
        params:
            ranches: Boolean indicating whether to group by ranch
            classes: Boolean indicating whether to group by class
            types: Boolean indicating whether to group by type
            varieties: Boolean indicating whether to group by variety
            include_actuals: Boolean indicating whether to include actual values in the output
        returns:
            pandas.DataFrame: DataFrame containing the metadata plus total predictions and actuals,
                            with both raw totals and per-hectare yields. If grouping parameters are True,
                            returns grouped and summed data.
        """
        group_cols = []
        if ranches:
            group_cols.append('Ranch')
        if classes:
            group_cols.append('Class') 
        if types is not None:
            group_cols.append('Type')
        if varieties:
            group_cols.append('Variety')

        #creating dataframe with metadata and hectares
        df = self.meta.copy()[group_cols + ['Ha']]

        #summing predictions
        for key, pred in self.predictions.items():
            df[key] = pred.sum(axis=1)

        if include_actuals:
            df['actuals'] = self.actuals.sum(axis=1)

        #grouping by specified parameters       
        if group_cols:
            df = df.groupby(group_cols).sum(numeric_only=True)
            
        else:
            #summing predictions
            df = df.sum(numeric_only=True)

        #calculating yield
        for key in self.predictions.keys():
            df[str(key) + '_yield'] = df[key] / df['Ha']

        if include_actuals:
            #calculating yield
            df['actuals_yield'] = df['actuals'] / df['Ha']
            
        return df

    def graph_ready(self,ranches=False, classes=False, types=False, varieties=False,include_actuals=True):
        """
        desc: Creates a dictionary of prediction matrices grouped by the specified grouping parameters.
        params:
            ranches: Boolean indicating whether to group by ranch
            classes: Boolean indicating whether to group by class
            types: Boolean indicating whether to group by type
            varieties: Boolean indicating whether to group by variety
            include_actuals: Boolean indicating whether to include actuals in the output
        returns:
            tuple: Contains:
                - int: Number of grouping columns used
                - dict: If no grouping parameters, contains three DataFrames under 'total' keys:
                       'total': Raw weekly predictions
                       'total_cumsum': Cumulative sum of predictions
                       'total_cumprop': Cumulative proportions
                       If grouping parameters provided, contains similar DataFrames for each group
                       with '_summed', '_summed_cumsum', and '_summed_cumprop' suffixes
                - pandas.Series: Hectares data, either total or grouped by the specified parameters
        """
        group_cols = []
        if ranches:
            group_cols.append('Ranch')
        if classes:
            group_cols.append('Class') 
        if types:
            group_cols.append('Type')
        if varieties:
            group_cols.append('Variety')

        if len(group_cols) == 0:
            #hectares data
            hectares = self.meta.agg({'Ha':'sum'})
            #summed predictions
            cats = {str(key) + '_sum': value.sum(axis=0) for key, value in self.predictions.items()}
            #adding summed actuals
            if include_actuals:
                cats['actuals_sum'] = self.actuals.sum(axis=0)
            #creating dataframe
            df = pd.DataFrame.from_dict(cats, orient='index')
            #cumulative sum
            df2 = df.cumsum(axis=1)
            #cumulative proportions
            df3 = df2.div(df2.iloc[:, -1], axis=0)
            
            return 3, {'total': df, 'total_cumsum': df2, 'total_cumprop': df3}, hectares
        
        else:
            #hectares data
            hectares = self.meta.groupby(group_cols).agg({'Ha':'sum'})
            #grouped indices
            group_keys = list(self.meta[group_cols].itertuples(index=False, name=None))
            group_series = pd.Series(group_keys)
            grouped_indices = group_series.groupby(group_series).groups
            #summed matrices
            summed_matrices = {}
            for key in self.predictions.keys():
                #summing predictions
                summed_matrix = {
                    group: self.predictions[key][list(indices)].sum(axis=0)
                    for group, indices in grouped_indices.items()
                }
                #creating dataframe
                result_df = pd.DataFrame.from_dict(summed_matrix, orient='index')
                result_df.index = pd.MultiIndex.from_tuples(result_df.index, names=group_cols)
                #cumulative sum 
                result_df2 = result_df.cumsum(axis=1)
                #cumulative proportions
                result_df3 = result_df2.div(result_df2.iloc[:, -1], axis=0)
                #adding to dictionary
                summed_matrices[str(key) + '_summed'] = result_df
                summed_matrices[str(key) + '_summed_cumsum'] = result_df2
                summed_matrices[str(key) + '_summed_cumprop'] = result_df3

            if include_actuals:
                #summing actuals
                actuals_matrix = {
                    group: self.actuals[list(indices)].sum(axis=0)
                    for group, indices in grouped_indices.items()
                }
                #creating dataframe
                actuals_df = pd.DataFrame.from_dict(actuals_matrix, orient='index')
                actuals_df.index = pd.MultiIndex.from_tuples(actuals_df.index, names=group_cols)
                #cumulative sum
                actuals_df2 = actuals_df.cumsum(axis=1)
                #cumulative proportions
                actuals_df3 = actuals_df2.div(actuals_df2.iloc[:, -1], axis=0)
                #adding to dictionary
                summed_matrices['actuals_summed'] = actuals_df
                summed_matrices['actuals_summed_cumsum'] = actuals_df2
                summed_matrices['actuals_summed_cumprop'] = actuals_df3

            return len(summed_matrices), summed_matrices, hectares


