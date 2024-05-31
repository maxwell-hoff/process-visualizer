import os
import pandas as pd
import numpy as np
from vispy import app, scene
from vispy.color import Color
import argparse

class BasicVispyVisualization:
    def __init__(self, data, case_spacing=10, team_spacing=20, dept_spacing=100, div_spacing=200, time_scale=1, height_scaling_factor=10):
        self.data = data
        self.case_spacing = case_spacing
        self.team_spacing = team_spacing
        self.dept_spacing = dept_spacing
        self.div_spacing = div_spacing
        self.time_scale = time_scale
        self.height_scaling_factor = height_scaling_factor
        self.data['X'] = self.data.apply(lambda row: self.calculate_position(row, 'Division', 'Department', 'Team'), axis=1)
        self.data['Y'] = self.data['Case Updated Date'].apply(lambda date: (pd.to_datetime(date) - pd.to_datetime('2000-01-01')).days / self.time_scale)
        self.data['Z'] = np.zeros(len(self.data))
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

    def on_hover(self, event):
        if event.is_dragging:
            return
        p = event.pos
        nearest, index = self.scatter.get_closest(p)
        if nearest is not None:
            info = self.data.iloc[index]
            self.text.text = f"Case ID: {info['Case ID']}, Status: {info['Case Status']}, Time: {info['Case Updated Date']}, Employee: {info['Employee ID']}"
            self.text.visible = True
            self.text.pos = p + 10
        else:
            self.text.visible = False

    def visualize(self):
        canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
        view = canvas.central_widget.add_view()

        x = self.data['X'].values
        y = self.data['Y'].values
        z = self.data['Z'].values

        positions = np.vstack([x, y, z]).T
        colors = [self.role_colors[role] for role in self.data['Role'].values]

        self.scatter = scene.visuals.Markers()
        self.scatter.set_data(positions, face_color=colors, size=5)
        view.add(self.scatter)

        axis = scene.visuals.XYZAxis(parent=view.scene)
        axis.transform = scene.transforms.MatrixTransform()
        axis.transform.scale([self.div_spacing, self.dept_spacing, self.team_spacing])
        axis.set_data(color='white')

        self.text = scene.visuals.Text('', color='white', anchor_x='left', parent=view.scene)
        self.text.visible = False

        view.camera = 'turntable'
        canvas.events.mouse_move.connect(self.on_hover)

        # Add arcs to connect statuses
        for case_id, case_data in self.data.groupby('Case ID'):
            case_data = case_data.sort_values('Case Updated Date')
            for i in range(len(case_data) - 1):
                start = [case_data.iloc[i]['X'], case_data.iloc[i]['Y'], 0]
                end = [case_data.iloc[i + 1]['X'], case_data.iloc[i + 1]['Y'], 0]
                duration = case_data.iloc[i + 1]['Time Since Last Modified']
                arc_height = duration / self.height_scaling_factor  # Scaling factor for arc height

                intermediate_point = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, arc_height]
                line_color = self.role_colors[case_data.iloc[i]['Role']]
                line = scene.visuals.Line(pos=np.array([start, intermediate_point, end]), color=line_color)
                view.add(line)

        # Add legend
        for i, (role, color) in enumerate(self.role_colors.items()):
            legend_text = scene.visuals.Text(f"{role}", color=color, pos=(5, 5 + i * 20), parent=canvas.scene)
        
        app.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize business process data")
    parser.add_argument('--data_path', type=str, default='data/business_process_data.csv', help="Path to the CSV file containing the data")
    parser.add_argument('--case_spacing', type=int, default=10, help="Spacing between cases")
    parser.add_argument('--team_spacing', type=int, default=20, help="Spacing between team members")
    parser.add_argument('--dept_spacing', type=int, default=100, help="Spacing between departments")
    parser.add_argument('--div_spacing', type=int, default=200, help="Spacing between divisions")
    parser.add_argument('--time_scale', type=int, default=1, help="Scale of the time axis")
    parser.add_argument('--height_scaling_factor', type=int, default=10, help="Scaling factor for arc height")
    
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    visualizer = BasicVispyVisualization(
        data
        , case_spacing=args.case_spacing
        , team_spacing=args.team_spacing
        , dept_spacing=args.dept_spacing
        , div_spacing=args.div_spacing
        , time_scale=args.time_scale
        , height_scaling_factor=args.height_scaling_factor
    )
    visualizer.visualize()