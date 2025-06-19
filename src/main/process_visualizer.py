import os
import pandas as pd
import numpy as np
from vispy import app, scene
from vispy.color import Color
import argparse
from vispy.util import keys

class CustomTurntablePanCamera(scene.cameras.TurntableCamera):
    """Turntable camera where RMB-drag pans (translates) instead of zooms."""

    def viewbox_mouse_event(self, event):  # noqa: C901
        """Custom mouse interaction: RMB pans instead of zoom."""

        if event.handled or not self.interactive:
            return

        # First, intercept our custom RMB-drag panning.
        if event.type == 'mouse_move' and 2 in event.buttons and not event.mouse_event.modifiers:
            if event.press_event is None:
                return

            # Perform translation similar to default Shift+LMB behaviour.
            norm = np.mean(self._viewbox.size)
            if self._event_value is None or len(self._event_value) == 2:
                self._event_value = self.center

            p1 = event.press_event.pos
            p2 = event.pos
            dist = (p1 - p2) / norm * self._scale_factor
            dist[1] *= -1

            dx, dy, dz = self._dist_to_trans(dist)
            ff = self._flip_factors
            up, forward, right = self._get_dim_vectors()
            dx, dy, dz = right * dx + forward * dy + up * dz
            dx, dy, dz = ff[0] * dx, ff[1] * dy, dz * ff[2]

            c = self._event_value
            self.center = c[0] + dx, c[1] + dy, c[2] + dz
            event.handled = True
            return  # Skip parent handling to avoid default zoom

        # For all other cases, fall back to the default TurntableCamera behaviour.
        super().viewbox_mouse_event(event)

class BasicVispyVisualization:
    def __init__(self, data, case_spacing=10, team_spacing=20, dept_spacing=100, div_spacing=200, time_scale=1, height_scaling_factor=10, enable_arcs=True):
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
        self.enable_arcs = enable_arcs and len(self.data) <= 20_000
        self.pan = False

    def calculate_position(self, row, primary, secondary, tertiary):
        base = hash(row[primary]) % self.div_spacing
        if secondary:
            base += (hash(row[secondary]) % self.dept_spacing)
        if tertiary:
            base += (hash(row[tertiary]) % self.team_spacing)
        return base

    def adjust_color_brightness(self, color, brightness_factor):
        color = np.array(color.rgb)
        if brightness_factor > 1.0:
            return color + (1 - color) * (brightness_factor - 1)
        else:
            return color * brightness_factor

    def on_hover(self, event):
        """Show tooltip of the data point under the cursor.

        The original implementation relied on an (old) MarkersVisual API
        providing a ``get_closest`` helper which is no longer available in
        modern VisPy releases. Rather than hard-crashing every time a mouse
        move occurs, we:

        1. First check whether the helper exists (legacy VisPy).
        2. If it doesn't, we simply skip the tooltip logic.  You still have
           the right-click pan and everything else, just no hover tooltip.  A
           robust picking implementation using ``MarkerPickingFilter`` could
           be added later if desired.
        """

        if event.is_dragging:
            return

        if not hasattr(self.scatter, "get_closest"):
            # API gone – silently ignore hover requests
            return

        p = event.pos
        try:
            nearest, index = self.scatter.get_closest(p)
        except Exception:
            # Any unexpected runtime issue, bail out without spamming errors
            return

        if nearest is not None:
            info = self.data.iloc[index]
            self.text.text = (
                f"Case ID: {info['Case ID']}, Status: {info['Case Status']}, "
                f"Time: {info['Case Updated Date']}, Employee: {info['Employee ID']}"
            )
            self.text.visible = True
            self.text.pos = p + 10
        else:
            self.text.visible = False

    def visualize(self):
        canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
        self.view = canvas.central_widget.add_view()

        x = self.data['X'].values
        y = self.data['Y'].values
        z = self.data['Z'].values

        positions = np.vstack([x, y, z]).T
        colors = []
        for _, row in self.data.iterrows():
            base_color = Color(self.role_colors[row['Role']])
            if row['Time Since Last Modified'] > row['Threshold']:
                color = self.adjust_color_brightness(base_color, 1.5)  # Brighter color
            else:
                color = self.adjust_color_brightness(base_color, 0.5)  # More transparent color
            colors.append(color)

        self.scatter = scene.visuals.Markers()
        self.scatter.set_data(positions, face_color=colors, size=5)
        self.view.add(self.scatter)

        axis = scene.visuals.XYZAxis(parent=self.view.scene)
        axis.transform = scene.transforms.MatrixTransform()
        axis.transform.scale([self.div_spacing, self.dept_spacing, self.team_spacing])
        axis.set_data(color='white')

        self.text = scene.visuals.Text('', color='white', anchor_x='left', parent=self.view.scene)
        self.text.visible = False

        self.view.camera = CustomTurntablePanCamera(up='z', fov=45)
        canvas.events.mouse_move.connect(self.on_hover)

        # Add arcs to connect statuses (optional – can be a performance hog)
        if self.enable_arcs:
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
                    self.view.add(line)

        # Add legend
        for i, (role, color) in enumerate(self.role_colors.items()):
            legend_text = scene.visuals.Text(f"{role}", color=color, pos=(10, 10 + i * 20), parent=canvas.scene)
            legend_text.font_size = 12  # Adjust font size if necessary

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
    parser.add_argument('--no_arcs', action='store_true', help="Skip drawing arcs (useful for large datasets)")
    
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
        , enable_arcs=not args.no_arcs
    )
    visualizer.visualize()