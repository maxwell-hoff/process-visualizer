import os
os.environ['VISPY_BACKEND'] = 'pyqt5'
from vispy import app, scene
import pandas as pd
import numpy as np

class BasicVispyVisualization:
    def __init__(self, data):
        self.data = data

    def visualize(self):
        canvas = scene.SceneCanvas(keys='interactive', show=True)
        view = canvas.central_widget.add_view()

        # Sample data for visualization (replace with actual data points)
        positions = np.random.normal(size=(1000, 3), scale=10)
        scatter = scene.visuals.Markers()
        scatter.set_data(positions, face_color='red', size=5)
        view.add(scatter)

        view.camera = 'turntable'
        app.run()

# Assuming business_data is already generated and available
if __name__ == '__main__':
    business_data = pd.read_csv('business_process_data.csv')
    visualizer = BasicVispyVisualization(business_data)
    visualizer.visualize()