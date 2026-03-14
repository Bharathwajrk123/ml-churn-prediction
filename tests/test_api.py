"""
Unit tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestHealth:
    """Tests for health check endpoint."""
    
    def test_health_check(self):
        """Test that the health check endpoint returns a 200 status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
        assert response.json()["status"] == "healthy"
    
    def test_health_check_model_loaded_field(self):
        """Test that the health check includes model_loaded field."""
        response = client.get("/health")
        assert "model_loaded" in response.json()


class TestPredict:
    """Tests for predict endpoint."""
    
    def test_predict_endpoint_exists(self):
        """Test that the predict endpoint is accessible."""
        response = client.post(
            "/predict",
            json={"features": {}}
        )
        # Should return 400 (bad request) because features are empty or invalid
        # But the endpoint should exist
        assert response.status_code in [400, 422, 503]
    
    def test_predict_with_valid_features(self):
        """Test prediction with valid features (if model is loaded)."""
        # Note: This test will fail if the model is not trained yet
        # It's designed to work when models/model.pkl exists
        features = {
            "tenure_months": 24,
            "monthly_charges": 65.5,
            "total_charges": 1570.0,
            "contract_one_year": 1,
        }
        response = client.post(
            "/predict",
            json={"features": features}
        )
        
        # Either success (200) if model loaded, or 503 if model not available
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability_churn" in data
            assert "probability_no_churn" in data
            assert 0 <= data["probability_churn"] <= 1
            assert 0 <= data["probability_no_churn"] <= 1
            assert data["prediction"] in [0, 1]
    
    def test_predict_response_schema(self):
        """Test that predict endpoint returns proper response schema."""
        # This test verifies the API structure without requiring a trained model
        response = client.post(
            "/predict",
            json={"features": {"test_feature": 1}}
        )
        
        # Model might not be loaded, but if it is, check schema
        if response.status_code == 200:
            data = response.json()
            # All these fields should be present
            expected_fields = {"prediction", "probability_churn", "probability_no_churn"}
            assert set(data.keys()) == expected_fields


class TestOpenAPI:
    """Tests for OpenAPI documentation."""
    
    def test_swagger_ui_accessible(self):
        """Test that Swagger UI is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_accessible(self):
        """Test that ReDoc is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_openapi_schema(self):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "openapi" in response.json()


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_predict_with_empty_features(self):
        """Test predict endpoint with empty features."""
        response = client.post(
            "/predict",
            json={"features": {}}
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 422, 503]
    
    def test_predict_with_missing_features(self):
        """Test predict with missing required fields."""
        response = client.post(
            "/predict",
            json={}
        )
        # Should return 422 (validation error) for missing features field
        assert response.status_code in [422, 400]
    
    def test_invalid_content_type(self):
        """Test endpoint with invalid content type."""
        response = client.post(
            "/predict",
            data="invalid data",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code in [422, 400, 415]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
