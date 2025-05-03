import unittest
import pandas as pd
import numpy as np
from src.table import CherryTable
import warnings

class TestCherryTable(unittest.TestCase):
    def setUp(self):
        # Suppress pandas bottleneck warning
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Create sample data for testing
        self.meta = pd.DataFrame({
            'Ranch': ['R1', 'R1', 'R2', 'R2'],
            'Class': ['C1', 'C2', 'C1', 'C2'],
            'Type': ['T1', 'T1', 'T2', 'T2'],
            'Variety': ['V1', 'V2', 'V1', 'V2']
        }, index=['P1', 'P2', 'P3', 'P4'])
        
        self.predictions = {
            'model1': np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            'model2': np.array([[2, 3], [4, 5], [6, 7], [8, 9]])
        }
        
        self.actuals = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.num_weeks = 2
        
        self.table = CherryTable(self.meta, self.predictions, self.actuals, self.num_weeks)

    def test_by_planting_id(self):
        # Test with valid planting IDs
        filtered = self.table.by_planting_id(['P1', 'P2'])
        self.assertIsNotNone(filtered)
        self.assertEqual(len(filtered.meta), 2)
        self.assertEqual(len(filtered.predictions['model1']), 2)
        self.assertEqual(len(filtered.actuals), 2)
        
        # Test with invalid planting IDs
        filtered = self.table.by_planting_id(['P5'])
        self.assertIsNone(filtered)
        
        # Test with mixed valid and invalid IDs
        filtered = self.table.by_planting_id(['P1', 'P5'])
        self.assertIsNone(filtered)

    def test_filter(self):
        # Test filtering by ranch
        filtered = self.table.filter(ranches=['R1'])
        self.assertIsNotNone(filtered)
        self.assertEqual(len(filtered.meta), 2)
        self.assertTrue(all(filtered.meta['Ranch'] == 'R1'))
        self.assertEqual(len(filtered.predictions['model1']), 2)
        self.assertEqual(len(filtered.actuals), 2)
        
        # Test filtering by multiple criteria
        filtered = self.table.filter(ranches=['R1'], classes=['C1'])
        self.assertIsNotNone(filtered)
        self.assertEqual(len(filtered.meta), 1)
        self.assertTrue(all(filtered.meta['Ranch'] == 'R1'))
        self.assertTrue(all(filtered.meta['Class'] == 'C1'))
        self.assertEqual(len(filtered.predictions['model1']), 1)
        self.assertEqual(len(filtered.actuals), 1)
        
        # Test filtering with no matches
        filtered = self.table.filter(ranches=['R3'])
        self.assertIsNone(filtered)

    def test_group_by(self):
        # Test grouping by ranch
        grouped = self.table.group_by(ranch=True)
        self.assertIsNotNone(grouped)
        self.assertEqual(len(grouped.meta), 2)  # Should have 2 groups (R1 and R2)
        self.assertEqual(len(grouped.predictions['model1']), 2)
        self.assertEqual(len(grouped.actuals), 2)
        
        # Test grouping by class
        grouped = self.table.group_by(class_=True)
        self.assertIsNotNone(grouped)
        self.assertEqual(len(grouped.meta), 2)  # Should have 2 groups (C1 and C2)
        self.assertEqual(len(grouped.predictions['model1']), 2)
        self.assertEqual(len(grouped.actuals), 2)
        
        # Test grouping by multiple columns
        grouped = self.table.group_by(ranch=True, class_=True)
        self.assertIsNotNone(grouped)
        self.assertEqual(len(grouped.meta), 4)  # Should have 4 groups (R1C1, R1C2, R2C1, R2C2)
        self.assertEqual(len(grouped.predictions['model1']), 4)
        self.assertEqual(len(grouped.actuals), 4)
        
        # Test summing entire dataset
        grouped = self.table.group_by(ranch=False, class_=False, type_=False, variety=False)
        self.assertIsNotNone(grouped)
        self.assertEqual(len(grouped.meta), 1)  # Should have 1 row with all data summed
        self.assertTrue(np.array_equal(grouped.predictions['model1'], np.array([[16, 20]])))
        self.assertTrue(np.array_equal(grouped.actuals, np.array([[16, 20]])))

if __name__ == '__main__':
    unittest.main() 