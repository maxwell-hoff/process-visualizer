import os
import pandas as pd
import numpy as np
from vispy import app, scene
from vispy.color import Color
import argparse

class BasicVispyVisualization:
    def __init__(self, data, case_spacing=10, team_spacing=20, dept_spacing=100, div_spacing=200):
        self.data = data
        self.case_spacing = case_spacing
        self.team_spacing = team_spacing
        self.dept_spacing = dept_spacing
        self.div_spacing = div_spacing
        self.data['X'] = self.data.apply(lambda row: self.calculate_position(row, 'Division', 'Department', 'Team'), axis=1)
        self.data['Y'] = self.data.apply(lambda row: self.calculate_position(row, 'Role', None, None), axis=1)
        self.role_colors = {
            'Manager': 'red',
            'Analyst': 'green',
            'Developer': 'blue',
            'Consultant': 'yellow',
            'Support': 'magenta'
        }

    def calculate_position(self, row, primary, secondary, tertiary):
        base = hash(row[primary]) % self.div_spacing
        if secondary:
            base += (hash(row[secondary]) % self.dept_spacing)
        if tertiary:
            base += (hash(row[tertiary]) % self.team_spacing)
        return base

    def visualize(self, scaling_factor=10 : int):
        canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
        view = canvas.central_widget.add_view()

        x = self.data['X'].values
        y = self.data['Y'].values
        z = np.zeros(len(self.data))

        positions = np.vstack([x, y, z]).T
        colors = [self.role_colors[role] for role in self.data['Role'].values]

        scatter = scene.visuals.Markers()
        scatter.set_data(positions, face_color=colors, size=5)
        view.add(scatter)

        axis = scene.visuals.XYZAxis(parent=view.scene)
        axis.transform = scene.transforms.MatrixTransform()
        axis.transform.scale([self.div_spacing, self.dept_spacing, self.team_spacing])
        axis.set_data(color='white')

        view.camera = 'turntable'

        # Add arcs to connect statuses
        for case_id, case_data in self.data.groupby('Case ID'):
            case_data = case_data.sort_values('Case Updated Date')
            for i in range(len(case_data) - 1):
                start = [case_data.iloc[i]['X'], case_data.iloc[i]['Y'], 0]
                end = [case_data.iloc[i + 1]['X'], case_data.iloc[i + 1]['Y'], 0]
                duration = case_data.iloc[i + 1]['Time Since Last Modified']
                arc_height = duration / scaling_factor  # Arbitrary scaling factor for arc height

                intermediate_point = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, arc_height]
                line_color = self.role_colors[case_data.iloc[i]['Role']]
                line = scene.visuals.Line(pos=np.array([start, intermediate_point, end]), color=line_color)
                view.add(line)

        app.run()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--case_spacing', type=int, default=20)
    argparser.add_argument('--team_spacing', type=int, default=20)
    argparser.add_argument('--dept_spacing', type=int, default=100)
    argparser.add_argument('--div_spacing', type=int, default=100)
    argparser.add_argument('--scaling_factor', type=float, default=10.0)
    args = argparser.parse_args()
    if not os.path.exists('data/business_process_data.csv'):
        raise FileNotFoundError('Business data not found. Please generate the data first.')
    business_data = pd.read_csv('data/business_process_data.csv')
    visualizer = BasicVispyVisualization(
        business_data
        , case_spacing=args.case_spacing
        , team_spacing=args.team_spacing
        , dept_spacing=args.dept_spacing
        , div_spacing=args.div_spacing
    )
    visualizer.visualize(scaling_factor=args.scaling_factor)