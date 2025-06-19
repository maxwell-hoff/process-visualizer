import pytest
import pandas as pd
import sys
sys.path.append('main/')
from create_data import BusinessProcessDataGenerator
from process_visualizer import BasicVispyVisualization

@pytest.fixture
def data_generator():
    return BusinessProcessDataGenerator(num_rows=1000)

@pytest.fixture
def sample_data(data_generator):
    return data_generator.generate()

def test_data_generation(sample_data):
    assert len(sample_data) > 1
    assert 'Employee ID' in sample_data.columns
    assert 'Division' in sample_data.columns
    assert 'Time Since Last Modified' in sample_data.columns

def test_vispy_visualization(sample_data):
    visualizer = BasicVispyVisualization(sample_data)
    assert visualizer.data is not None
    # This is a placeholder check since actual rendering can't be verified in a test easily
    assert len(visualizer.data) > 1