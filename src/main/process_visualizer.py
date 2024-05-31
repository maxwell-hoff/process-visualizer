import os
import pandas as pd
import numpy as np
from vispy import app, scene
from vispy.color import Color

class BasicVispyVisualization:
    def __init__(self, data):
        self.data = data
        self.data['X'] = self.data.apply(lambda row: self.calculate_position(row, 'Division', 'Department', 'Team'), axis=1)
        self.data['Z'] = self.data.apply(lambda row: self.calculate_position(row, 'Role', None, None), axis=1)
        self.role_colors = {
            'Manager': 'red',
            'Analyst': 'green',
            'Developer': 'blue',
            'Consultant': 'yellow',
            'Support': 'magenta'
        }
        self.num_rows = len(self.data)

    def calculate_position(self, row, primary, secondary, tertiary):
        base = hash(row[primary]) % 1000
        if secondary:
            base += (hash(row[secondary]) % 100) * 10
        if tertiary:
            base += hash(row[tertiary]) % 10
        return base

    def visualize(self):
        canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
        view = canvas.central_widget.add_view()

        x = self.data['X'].values
        y = np.zeros(len(self.data))
        z = self.data['Z'].values

        positions = np.vstack([x, y, z]).T
        colors = [self.role_colors[role] for role in self.data['Role'].values]

        scatter = scene.visuals.Markers()
        scatter.set_data(positions, face_color=colors, size=5)
        view.add(scatter)

        axis = scene.visuals.XYZAxis(parent=view.scene)
        axis.transform = scene.transforms.MatrixTransform()
        axis.transform.scale([1000, 1000, 1000])

        view.camera = 'turntable'

        # Add arcs to connect statuses
        counter = 0
        for case_id, case_data in self.data.groupby('Case ID'):
            case_data = case_data.sort_values('Case Updated Date')
            for i in range(len(case_data) - 1):
                counter+=1
                start = [case_data.iloc[i]['X'], 0, case_data.iloc[i]['Z']]
                end = [case_data.iloc[i + 1]['X'], 0, case_data.iloc[i + 1]['Z']]
                duration = case_data.iloc[i + 1]['Time Since Last Modified']
                arc_height = duration / 10.0  # Arbitrary scaling factor for arc height
                print(f"height: {arc_height}, perc compl: {counter/self.num_rows}, row: {counter}, num_rows: {self.num_rows}")

                intermediate_point = [(start[0] + end[0]) / 2, arc_height, (start[2] + end[2]) / 2]
                line = scene.visuals.Line(pos=np.array([start, intermediate_point, end]), color='white')
                view.add(line)

        app.run()

# Assuming business_data is already generated and available
if __name__ == '__main__':
    if not os.path.exists('data/business_process_data.csv'):
        raise FileNotFoundError('Business data not found. Please generate the data first.')
    business_data = pd.read_csv('data/business_process_data.csv')
    visualizer = BasicVispyVisualization(business_data)
    visualizer.visualize()