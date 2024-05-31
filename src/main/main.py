from create_data import BusinessProcessDataGenerator
from process_visualizer import BasicVispyVisualization
import argparse

class RunVisualizer:
    def __init__(self):
        pass
        generator = BusinessProcessDataGenerator()
    
    def generate_date(self):
        business_data = generator.generate()
        
    def execute(self):
        
        visualizer = BasicVispyVisualization(business_data)
        visualizer.visualize()

    # Load data
    business_data = pd.read_csv('/path/to/your/generated/data.csv')
    visualizer = BasicVispyVisualization(business_data)
    visualizer.visualize()