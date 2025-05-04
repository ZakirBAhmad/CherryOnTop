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
        filtered = self.table.filter(ranch_list=['R1'])
        self.assertIsNotNone(filtered)
        self.assertEqual(len(filtered.meta), 2)
        self.assertTrue(all(filtered.meta['Ranch'] == 'R1'))
        self.assertEqual(len(filtered.predictions['model1']), 2)
        self.assertEqual(len(filtered.actuals), 2)
        
        # Test filtering by multiple criteria
        filtered = self.table.filter(ranch_list=['R1'], class_list=['C1'])
        self.assertIsNotNone(filtered)
        self.assertEqual(len(filtered.meta), 1)
        self.assertTrue(all(filtered.meta['Ranch'] == 'R1'))
        self.assertTrue(all(filtered.meta['Class'] == 'C1'))
        self.assertEqual(len(filtered.predictions['model1']), 1)
        self.assertEqual(len(filtered.actuals), 1)
        
        # Test filtering with no matches
        filtered = self.table.filter(ranch_list=['R3'])
        self.assertIsNone(filtered)

    def test_summary(self):
        # Test summary with no grouping
        summary = self.table.summary()
        self.assertIsNotNone(summary)
        self.assertTrue(isinstance(summary, pd.Series))
        
        # Test summary with ranch grouping
        summary = self.table.summary(ranches=True)
        self.assertIsNotNone(summary)
        self.assertTrue(isinstance(summary, pd.DataFrame))
        self.assertEqual(len(summary), 2)  # Should have 2 groups (R1 and R2)
        
        # Test summary with class grouping
        summary = self.table.summary(classes=True)
        self.assertIsNotNone(summary)
        self.assertTrue(isinstance(summary, pd.DataFrame))
        self.assertEqual(len(summary), 2)  # Should have 2 groups (C1 and C2)
        
        # Test summary with multiple grouping
        summary = self.table.summary(ranches=True, classes=True)
        self.assertIsNotNone(summary)
        self.assertTrue(isinstance(summary, pd.DataFrame))
        self.assertEqual(len(summary), 4)  # Should have 4 groups (R1C1, R1C2, R2C1, R2C2)

if __name__ == '__main__':
    unittest.main() 