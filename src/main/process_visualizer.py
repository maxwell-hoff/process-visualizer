import os
import math
import random
import pandas as pd
import numpy as np
from vispy import app, scene
from vispy.color import Color
import argparse
import time
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
    def __init__(self, data, anim_duration=20, hierarchy=None, height_scaling_factor=10, enable_arcs=True):
        self.data = data

        # Hierarchy definition (outer → inner)
        if hierarchy is None:
            hierarchy = ['Division', 'Department', 'Team', 'Employee ID']
        self.hierarchy = hierarchy

        self.anim_duration = anim_duration  # seconds
        self.height_scaling_factor = height_scaling_factor

        # Compute hierarchical nested positions
        self._assign_nested_positions()

        self.data['__ts__'] = pd.to_datetime(self.data['Case Updated Date']).astype('int64') // 1_000_000_000  # to seconds
        self.start_ts = self.data['__ts__'].min()
        self.end_ts = self.data['__ts__'].max()
        self.role_colors = {
            'Manager': 'red',
            'Analyst': 'green',
            'Developer': 'blue',
            'Consultant': 'yellow',
            'Support': 'magenta'
        }
        self.enable_arcs = enable_arcs and len(self.data) <= 20_000
        self.paused = False
        self.current_ts = self.start_ts
        self.pan = False

    # ------------------------------------------------------------------
    # Hierarchical layout helpers
    # ------------------------------------------------------------------

    def _assign_nested_positions(self):
        """Assign X, Y coordinates to each row based on the hierarchy."""

        # Build nested dictionary representing the hierarchy tree
        tree = {}
        for _, row in self.data.iterrows():
            node = tree
            for level in self.hierarchy:
                key = row[level]
                node = node.setdefault(key, {})

        # Recursively assign rectangles
        rects = {}  # mapping from path tuple to rectangle (x0,y0,x1,y1)

        def subdivide(node, rect, path):
            """Recursively allocate rectangles to children."""
            children = list(node.keys())
            if not children:
                return
            n = len(children)
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)

            x0, y0, x1, y1 = rect
            w = (x1 - x0) / cols
            h = (y1 - y0) / rows

            for idx, child in enumerate(children):
                c_col = idx % cols
                c_row = idx // cols
                child_rect = (
                    x0 + c_col * w,
                    y0 + c_row * h,
                    x0 + (c_col + 1) * w,
                    y0 + (c_row + 1) * h,
                )
                child_path = path + (child,)
                rects[child_path] = child_rect
                subdivide(node[child], child_rect, child_path)

        # Start subdivision with the whole [0,1]x[0,1] square.
        subdivide(tree, (0.0, 0.0, 1.0, 1.0), tuple())

        # Now assign each row a coordinate inside its employee rectangle path
        xs = []
        ys = []
        rng = random.Random(123)
        for _, row in self.data.iterrows():
            # Build path through the hierarchy *excluding the last level* (Employee ID)
            path = tuple(row[level] for level in self.hierarchy[:-1])
            leaf_rect = rects[path]
            x0, y0, x1, y1 = leaf_rect
            # slightly inset to avoid overlap
            inset = 0.02 * min(x1 - x0, y1 - y0)
            x0 += inset
            y0 += inset
            x1 -= inset
            y1 -= inset
            xs.append(rng.uniform(x0, x1))
            ys.append(rng.uniform(y0, y1))

        self.data['X'] = xs
        self.data['Y'] = ys

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
        z = np.zeros(len(self.data))

        positions = np.vstack([x, y, z]).T
        colors = []
        for _, row in self.data.iterrows():
            base_color = Color(self.role_colors[row['Role']])
            if row['Time Since Last Modified'] > row['Threshold']:
                color = self.adjust_color_brightness(base_color, 1.5)  # Brighter color
            else:
                color = self.adjust_color_brightness(base_color, 0.5)  # More transparent color
            colors.append(color)

        self.positions = positions
        self.colors = np.array(colors)

        self.scatter = scene.visuals.Markers()
        # Initialize with a single invisible point to avoid zero-size arrays
        dummy_pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
        dummy_color = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=float)
        self.scatter.set_data(dummy_pos, face_color=dummy_color, size=5)
        self.view.add(self.scatter)

        axis = scene.visuals.XYZAxis(parent=self.view.scene)
        axis.transform = scene.transforms.MatrixTransform()
        span_x = max(x) - min(x)
        span_y = max(y) - min(y)
        axis.transform.scale([span_x if span_x else 1, span_y if span_y else 1, 1])
        axis.set_data(color='white')

        self.text = scene.visuals.Text('', color='white', anchor_x='left', parent=self.view.scene)
        self.text.visible = False

        self.view.camera = CustomTurntablePanCamera(up='z', fov=45)
        canvas.events.mouse_move.connect(self.on_hover)

        # Key controls
        canvas.events.key_press.connect(self.on_key_press)

        # Animation timer
        self.t0 = time.perf_counter()
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

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

    # ---------------- Animation handlers -----------------

    def _update_scatter(self):
        """Update scatter visual to display points up to current_ts."""
        mask = self.data['__ts__'].values <= self.current_ts
        if mask.any():
            self.scatter.visible = True
            self.scatter.set_data(self.positions[mask], face_color=self.colors[mask], size=5)
        else:
            self.scatter.visible = False

    def on_timer(self, event):
        if self.paused:
            return
        elapsed = time.perf_counter() - self.t0
        frac = (elapsed % self.anim_duration) / self.anim_duration
        self.current_ts = self.start_ts + frac * (self.end_ts - self.start_ts)
        self._update_scatter()

    def on_key_press(self, event):
        if event.key == keys.SPACE:
            self.paused = not self.paused
            if not self.paused:
                # resume timeline origin to maintain continuity
                self.t0 = time.perf_counter() - (self.current_ts - self.start_ts) * self.anim_duration / (self.end_ts - self.start_ts)
        elif event.key == keys.RIGHT:
            step = (self.end_ts - self.start_ts) * 0.01  # 1% step
            self.current_ts = min(self.current_ts + step, self.end_ts)
            self._update_scatter()
        elif event.key == keys.LEFT:
            step = (self.end_ts - self.start_ts) * 0.01
            self.current_ts = max(self.current_ts - step, self.start_ts)
            self._update_scatter()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize business process data")
    parser.add_argument('--data_path', type=str, default='data/business_process_data.csv', help="Path to the CSV file containing the data")
    parser.add_argument('--anim_duration', type=int, default=20, help="Duration of the animation")
    parser.add_argument('--height_scaling_factor', type=int, default=10, help="Scaling factor for arc height")
    parser.add_argument('--no_arcs', action='store_true', help="Skip drawing arcs (useful for large datasets)")
    
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    visualizer = BasicVispyVisualization(
        data
        , anim_duration=args.anim_duration
        , height_scaling_factor=args.height_scaling_factor
        , enable_arcs=not args.no_arcs
    )
    visualizer.visualize()