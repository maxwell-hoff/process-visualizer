import os
import pandas as pd
import numpy as np
from vispy import app, scene

class BasicVispyVisualization:
    def __init__(self, data):
        self.data = data
        self.data['X'] = self.data.apply(lambda row: self.calculate_position(row, 'Division', 'Department', 'Team'), axis=1)
        self.data['Z'] = self.data.apply(lambda row: self.calculate_position(row, 'Role', None, None), axis=1)

    def calculate_position(self, row, primary, secondary, tertiary):
        base = hash(row[primary]) % 1000
        if secondary:
            base += (hash(row[secondary]) % 100) * 10
        if tertiary:
            base += hash(row[tertiary]) % 10
        return base

    def visualize(self):
        canvas = scene.SceneCanvas(keys='interactive', show=True)
        view = canvas.central_widget.add_view()

        x = self.data['X'].values
        y = np.zeros(len(self.data))
        z = self.data['Z'].values

        positions = np.vstack([x, y, z]).T
        scatter = scene.visuals.Markers()
        scatter.set_data(positions, face_color='red', size=5)
        view.add(scatter)

        axis = scene.visuals.XYZAxis(parent=view.scene)

        view.camera = 'turntable'
        app.run()

# Assuming business_data is already generated and available
if __name__ == '__main__':
    if not os.path.exists('data/business_process_data.csv'):
        raise FileNotFoundError('Business data not found. Please generate the data first.')
    business_data = pd.read_csv('data/business_process_data.csv')
    visualizer = BasicVispyVisualization(business_data)
    visualizer.visualize()