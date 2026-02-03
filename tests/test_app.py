"""Tests for the Flask application."""
import pytest
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    """Test that home page loads successfully."""
    response = client.get('/')
    assert response.status_code == 200


def test_metrics_endpoint(client):
    """Test that metrics endpoint returns data."""
    response = client.get('/metrics')
    assert response.status_code == 200


def test_predict_without_model(client):
    """Test predict endpoint returns error when model not loaded."""
    response = client.post('/predict',
                          data=json.dumps([{'eta': 0.1, 'f': 0.5, 'f_prime': 1.0}]),
                          content_type='application/json')
    # Should return 500 if model not loaded
    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data


def test_predict_missing_columns(client):
    """Test predict endpoint with missing columns."""
    response = client.post('/predict',
                          data=json.dumps([{'eta': 0.1}]),  # Missing f and f_prime
                          content_type='application/json')
    # Should return error (either 400 for missing columns or 500 for no model)
    assert response.status_code in [400, 500]


def test_model_info_endpoint(client):
    """Test model-info endpoint."""
    response = client.get('/model-info')
    # Should return 500 if model not loaded, or 200 with model info
    assert response.status_code in [200, 500]


def test_retrain_status_endpoint(client):
    """Test retrain-status endpoint."""
    response = client.get('/retrain-status')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'is_retraining' in data
