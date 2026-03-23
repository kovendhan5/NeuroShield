from flask import Flask, jsonify, request
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps
import time
import random
import os
import logging
from datetime import datetime
import psycopg2
from redis import Redis
import json

# ===== SETUP =====
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database
db_url = os.getenv('DATABASE_URL', 'postgresql://admin:neuroshield_db_pass@postgres:5432/neuroshield_db')
redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')

# Metrics
requests_total = Counter('app_requests_total', 'Total requests', ['method', 'endpoint'])
request_latency = Histogram('app_request_latency_seconds', 'Request latency', ['endpoint'])
request_errors = Counter('app_request_errors_total', 'Total request errors', ['endpoint', 'error_type'])
app_health = Gauge('app_health_percentage', 'Application health percentage')
db_connections = Gauge('db_connections_active', 'Active database connections')
processed_jobs = Counter('jobs_processed_total', 'Total jobs processed', ['status'])
cache_hits = Counter('cache_hits_total', 'Total cache hits')
cache_misses = Counter('cache_misses_total', 'Total cache misses')

# Global state
app.health_pct = 100
app.failure_mode = False

# ===== MIDDLEWARE =====
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        lat = time.time() - request.start_time
        request_latency.labels(endpoint=request.endpoint or 'unknown').observe(lat)

    requests_total.labels(method=request.method, endpoint=request.endpoint or 'unknown').inc()
    return response

# ===== ROUTES =====

# Health checks
@app.route('/health', methods=['GET'])
def health_simple():
    """Simple health check"""
    app_health.set(app.health_pct)
    status = 'healthy' if app.health_pct >= 50 else 'degraded' if app.health_pct > 0 else 'unhealthy'
    return jsonify({'status': status, 'health': app.health_pct}), 200 if status == 'healthy' else 503

@app.route('/health/detailed', methods=['GET'])
def health_detailed():
    """Detailed health check"""
    health_info = {
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'healthy' if app.health_pct >= 80 else 'degraded',
        'health_percentage': app.health_pct,
        'services': {
            'database': check_database(),
            'cache': check_cache(),
            'api': 'up'
        },
        'metrics': {
            'uptime': time.time(),
            'requests_processed': int(requests_total._value.get()),
            'errors_total': int(request_errors._value.get())
        }
    }
    app_health.set(app.health_pct)
    return jsonify(health_info), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    app_health.set(app.health_pct)
    db_connections.set(count_db_connections())
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

# API endpoints
@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """Get job list from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, status, created_at FROM jobs LIMIT 50')
        jobs = cursor.fetchall()
        cursor.close()
        conn.close()

        cache_hits.inc()
        processed_jobs.labels(status='success').inc()

        return jsonify({
            'total': len(jobs),
            'jobs': [{'id': j[0], 'name': j[1], 'status': j[2], 'created_at': str(j[3])} for j in jobs]
        }), 200
    except Exception as e:
        logger.error(f'Jobs list error: {e}')
        request_errors.labels(endpoint='list_jobs', error_type=type(e).__name__).inc()
        processed_jobs.labels(status='error').inc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs', methods=['POST'])
def create_job():
    """Create a new job"""
    try:
        data = request.json
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            'INSERT INTO jobs (name, status, created_at) VALUES (%s, %s, NOW()) RETURNING id',
            (data.get('name', 'unnamed'), 'pending')
        )
        job_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()

        processed_jobs.labels(status='created').inc()
        return jsonify({'id': job_id, 'status': 'created'}), 201
    except Exception as e:
        logger.error(f'Job create error: {e}')
        request_errors.labels(endpoint='create_job', error_type=type(e).__name__).inc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_data():
    """Process data (CPU intensive)"""
    try:
        # Simulate processing
        time.sleep(random.uniform(0.1, 2.0))

        # Random chance to fail
        if app.failure_mode and random.random() < 0.3:
            raise Exception('Processing failed - simulated error')

        processed_jobs.labels(status='success').inc()
        return jsonify({'result': 'processed', 'timestamp': datetime.utcnow().isoformat()}), 200
    except Exception as e:
        logger.error(f'Processing error: {e}')
        request_errors.labels(endpoint='process', error_type='processing_error').inc()
        processed_jobs.labels(status='failed').inc()
        app.health_pct = max(0, app.health_pct - 10)
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/<key>', methods=['GET'])
def get_cache(key):
    """Get value from cache"""
    try:
        redis = Redis.from_url(redis_url)
        value = redis.get(f'app:{key}')

        if value:
            cache_hits.inc()
            return jsonify({'key': key, 'value': json.loads(value), 'source': 'cache'}), 200
        else:
            cache_misses.inc()
            return jsonify({'key': key, 'value': None}), 404
    except Exception as e:
        logger.error(f'Cache error: {e}')
        request_errors.labels(endpoint='get_cache', error_type=type(e).__name__).inc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/<key>', methods=['POST'])
def set_cache(key):
    """Set value in cache"""
    try:
        data = request.json
        redis = Redis.from_url(redis_url)
        redis.setex(f'app:{key}', 3600, json.dumps(data))
        return jsonify({'key': key, 'status': 'cached'}), 201
    except Exception as e:
        logger.error(f'Cache set error: {e}')
        request_errors.labels(endpoint='set_cache', error_type=type(e).__name__).inc()
        return jsonify({'error': str(e)}), 500

# Status endpoints
@app.route('/api/status', methods=['GET'])
def system_status():
    """Get system status"""
    return jsonify({
        'app_name': 'NeuroShield Microservice',
        'version': '1.0.0',
        'health': app.health_pct,
        'failure_mode': app.failure_mode,
        'timestamp': datetime.utcnow().isoformat(),
        'uptime': time.time()
    }), 200

@app.route('/api/status/degraded', methods=['POST'])
def set_degraded():
    """Simulate degradation (for testing)"""
    app.health_pct = 30
    app.failure_mode = True
    logger.warning('Status set to degraded - simulating failure')
    return jsonify({'status': 'degraded', 'health': app.health_pct}), 200

@app.route('/api/status/healthy', methods=['POST'])
def set_healthy():
    """Restore to healthy (for healing verification)"""
    app.health_pct = 100
    app.failure_mode = False
    logger.info('Status restored to healthy')
    return jsonify({'status': 'healthy', 'health': app.health_pct}), 200

# ===== HELPER FUNCTIONS =====

def get_db_connection():
    """Get database connection"""
    conn = psycopg2.connect(db_url)
    return conn

def init_database():
    """Initialize database schema"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255),
                status VARCHAR(50),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        ''')

        conn.commit()
        cursor.close()
        conn.close()
        logger.info('Database initialized')
    except Exception as e:
        logger.error(f'Database init error: {e}')

def count_db_connections():
    """Count active connections"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM pg_stat_activity")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    except:
        return 0

def check_database():
    """Check database health"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT NOW()')
        cursor.close()
        conn.close()
        return 'healthy'
    except:
        return 'degraded'

def check_cache():
    """Check cache health"""
    try:
        redis = Redis.from_url(redis_url)
        redis.ping()
        return 'healthy'
    except:
        return 'degraded'

# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    request_errors.labels(endpoint='unknown', error_type='server_error').inc()
    return jsonify({'error': 'Internal server error'}), 500

# ===== STARTUP =====

if __name__ == '__main__':
    init_database()
    logger.info('NeuroShield Microservice starting...')
    app.run(host='0.0.0.0', port=5000, debug=False)
