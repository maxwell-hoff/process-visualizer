from create_data import BusinessProcessDataGenerator
from process_visualizer import BasicVispyVisualization
import argparse
import pandas as pd

class RunVisualizer:
    def __init__(self):
        pass
        self.generator = BusinessProcessDataGenerator()
    
    def generate_date(self):
        self.business_data = self.generator.generate()
        
    def execute(self):
        
        visualizer = BasicVispyVisualization(self.business_data)
        visualizer.visualize()

    # Load data
    business_data = pd.read_csv('/path/to/your/generated/data.csv')
    visualizer = BasicVispyVisualization(business_data)
    visualizer.visualize()