"""
Unit tests for telemetry collector.
Run with: pytest tests/test_telemetry.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.telemetry.collector import (
    TelemetryCollector, JenkinsPoll, PrometheusPoll, TelemetryData
)


class TestJenkinsPoll:
    """Test Jenkins polling."""
    
    @patch('requests.Session.get')
    def test_get_last_build_status_success(self, mock_get):
        """Test successful build status fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'result': 'SUCCESS',
            'duration': 5000
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        jenkins = JenkinsPoll('http://localhost:8080')
        result = jenkins.get_last_build_status('test-job')
        
        assert result['status'] == 'SUCCESS'
        assert result['duration_ms'] == 5000
    
    @patch('requests.Session.get')
    def test_get_last_build_status_failure(self, mock_get):
        """Test build status fetch failure."""
        mock_get.side_effect = Exception("Connection failed")
        
        jenkins = JenkinsPoll('http://localhost:8080')
        result = jenkins.get_last_build_status('test-job')
        
        assert result is None
    
    @patch('requests.Session.get')
    def test_get_queue_length(self, mock_get):
        """Test queue length fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'items': [{'id': 1}, {'id': 2}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        jenkins = JenkinsPoll('http://localhost:8080')
        result = jenkins.get_queue_length()
        
        assert result == 2


class TestPrometheusPoll:
    """Test Prometheus polling."""
    
    @patch('requests.Session.get')
    def test_query_success(self, mock_get):
        """Test successful Prometheus query."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': {
                'result': [{'value': [1234567890, '0.42']}]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        prometheus = PrometheusPoll('http://localhost:9090')
        result = prometheus.query('test_metric')
        
        assert len(result) == 1
        assert result[0]['value'] == [1234567890, '0.42']
    
    @patch('requests.Session.get')
    def test_get_cpu_usage(self, mock_get):
        """Test CPU usage fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': {
                'result': [{'value': [1234567890, '0.5']}]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        prometheus = PrometheusPoll('http://localhost:9090')
        result = prometheus.get_cpu_usage()
        
        assert result == 50.0


class TestTelemetryData:
    """Test TelemetryData dataclass."""
    
    def test_telemetry_data_creation(self):
        """Test TelemetryData object creation."""
        data = TelemetryData(
            timestamp='2025-01-01T00:00:00',
            jenkins_last_build_status='SUCCESS',
            jenkins_last_build_duration=5000.0,
            jenkins_queue_length=2,
            prometheus_cpu_usage=45.5,
            prometheus_memory_usage=60.0,
            prometheus_pod_count=10,
            prometheus_error_rate=0.001
        )
        
        assert data.timestamp == '2025-01-01T00:00:00'
        assert data.jenkins_last_build_status == 'SUCCESS'
        assert data.prometheus_cpu_usage == 45.5


class TestTelemetryCollector:
    """Test TelemetryCollector orchestration."""
    
    @patch('src.telemetry.collector.JenkinsPoll')
    @patch('src.telemetry.collector.PrometheusPoll')
    def test_collector_initialization(self, mock_prometheus, mock_jenkins):
        """Test collector initialization."""
        collector = TelemetryCollector(
            jenkins_url='http://localhost:8080',
            prometheus_url='http://localhost:9090',
            output_csv='/tmp/test.csv'
        )
        
        assert collector.jenkins_job == 'default-job'
        assert collector.poll_interval == 10
    
    @patch('src.telemetry.collector.JenkinsPoll')
    @patch('src.telemetry.collector.PrometheusPoll')
    def test_collect_once(self, mock_prometheus_class, mock_jenkins_class):
        """Test single telemetry collection."""
        mock_jenkins = Mock()
        mock_jenkins.get_last_build_status.return_value = {
            'status': 'SUCCESS',
            'duration_ms': 5000
        }
        mock_jenkins.get_queue_length.return_value = 2
        mock_jenkins_class.return_value = mock_jenkins
        
        mock_prometheus = Mock()
        mock_prometheus.get_cpu_usage.return_value = 45.5
        mock_prometheus.get_memory_usage.return_value = 60.0
        mock_prometheus.get_pod_count.return_value = 10
        mock_prometheus.get_error_rate.return_value = 0.001
        mock_prometheus_class.return_value = mock_prometheus
        
        collector = TelemetryCollector(
            jenkins_url='http://localhost:8080',
            prometheus_url='http://localhost:9090',
            output_csv='/tmp/test.csv'
        )
        
        data = collector.collect_once()
        
        assert data.jenkins_last_build_status == 'SUCCESS'
        assert data.prometheus_cpu_usage == 45.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
