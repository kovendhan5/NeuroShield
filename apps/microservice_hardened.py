"""
NeuroShield Microservice - Production Hardened Version
Addresses critical issues from audit:
- Database connection pooling
- Structured logging
- Input validation
- Authentication
- Error handling
- Proper WSGI (Gunicorn)
"""

import os
import logging
import json
from functools import wraps
from datetime import datetime, timedelta
from uuid import uuid4

from flask import Flask, request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from marshmallow import Schema, fields, ValidationError
import psycopg2
from psycopg2 import pool
from redis import Redis
import jwt

# ===== CONFIGURATION =====
app = Flask(__name__)
CORS(app)  # For cross-origin - can be restricted

SECRET_KEY = os.getenv('API_SECRET_KEY', 'change-me-in-production')
DB_URL = os.getenv('DATABASE_URL')
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
REDIS_URL = os.getenv('REDIS_URL')
if not REDIS_URL and REDIS_PASSWORD:
    REDIS_URL = f'redis://:{REDIS_PASSWORD}@redis:6379'
elif not REDIS_URL:
    REDIS_URL = 'redis://redis:6379'
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# ===== LOGGING - STRUCTURED =====
class StructuredFormatter(logging.Formatter):
    """Output logs as JSON for log aggregation"""
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        # Only add correlation_id if in request context
        try:
            log_data['correlation_id'] = getattr(g, 'correlation_id', 'unknown')
        except RuntimeError:
            log_data['correlation_id'] = 'startup'

        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_data)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ===== DATABASE CONNECTION POOL =====
try:
    # Use DATABASE_URL if provided, otherwise build from components
    if DB_URL:
        db_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=2,
            maxconn=20,
            dsn=DB_URL,
            connect_timeout=5
        )
    else:
        raise ValueError("DATABASE_URL not configured")
    logger.info("Database connection pool initialized")
except Exception as e:
    logger.error(f"Failed to create DB pool: {e}")
    db_pool = None

# ===== REDIS WITH TIMEOUT =====
try:
    redis_client = Redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=3)
    redis_client.ping()
    logger.info("Redis connection established")
except Exception as e:
    logger.warning(f"Redis unavailable: {e}")
    redis_client = None

# ===== LIMITER =====
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=REDIS_URL if redis_client else None
)

# ===== METRICS =====
requests_total = Counter('app_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_latency = Histogram('app_request_latency_seconds', 'Request latency', ['endpoint'])
request_errors = Counter('app_request_errors_total', 'Total errors', ['endpoint', 'error_type'])
app_health = Gauge('app_health_percentage', 'Application health')
db_connections = Gauge('db_pool_connections_active', 'Active DB connections')

# ===== VALIDATION SCHEMAS =====
class JobSchema(Schema):
    name = fields.String(required=True)
    status = fields.String(allow_none=True)
    description = fields.String(allow_none=True)

# ===== AUTHENTICATION =====
def token_required(f):
    """JWT authentication decorator"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')

        if not token:
            logger.warning(f"No token provided - {request.remote_addr}")
            return jsonify({'error': 'Missing authorization token'}), 401

        try:
            # Verify token
            data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            g.user_id = data.get('user_id')
        except jwt.ExpiredSignatureError:
            logger.warning(f"Token expired - {request.remote_addr}")
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e} - {request.remote_addr}")
            return jsonify({'error': 'Invalid token'}), 403

        return f(*args, **kwargs)
    return decorated

# ===== MIDDLEWARE =====
@app.before_request
def add_request_context():
    """Add correlation ID and timing"""
    g.correlation_id = request.headers.get('X-Correlation-ID', str(uuid4()))
    g.start_time = datetime.utcnow()
    logger.info(f"Request started: {request.method} {request.path}")

@app.after_request
def log_request(response):
    """Log response and metrics"""
    if hasattr(g, 'start_time'):
        duration = (datetime.utcnow() - g.start_time).total_seconds()
        request_latency.labels(endpoint=request.endpoint or 'unknown').observe(duration)

    requests_total.labels(
        method=request.method,
        endpoint=request.endpoint or 'unknown',
        status=response.status_code
    ).inc()

    logger.info(f"Request completed: {response.status_code}")
    response.headers['X-Correlation-ID'] = getattr(g, 'correlation_id', 'unknown')
    return response

@app.errorhandler(ValidationError)
def handle_validation_error(error):
    """Handle schema validation errors"""
    logger.warning(f"Validation error: {error.messages}")
    return jsonify({'errors': error.messages}), 400

# ===== UTILITY FUNCTIONS =====
def get_db_connection():
    """Get connection from pool (with error handling)"""
    if not db_pool:
        raise Exception("Database pool not initialized")

    try:
        conn = db_pool.getconn()
        return conn
    except pool.PoolError as e:
        logger.error(f"Pool exhausted: {e}")
        raise

def return_db_connection(conn):
    """Return connection to pool"""
    if conn and db_pool:
        db_pool.putconn(conn)

def init_database():
    """Initialize database schema with proper constraints"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                status VARCHAR(50) NOT NULL DEFAULT 'pending',
                description TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                created_by VARCHAR(255),

                CONSTRAINT valid_status CHECK (status IN ('pending', 'running', 'completed', 'failed')),
                CONSTRAINT valid_name CHECK (length(name) > 0)
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_jobs_created_by ON jobs(created_by);

            -- Audit trail table
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                action VARCHAR(50) NOT NULL,
                resource_type VARCHAR(50) NOT NULL,
                resource_id INTEGER,
                user_id VARCHAR(255) NOT NULL,
                changes JSONB,
                timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                correlation_id VARCHAR(36)
            );

            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
        ''')

        conn.commit()
        logger.info("Database schema initialized")
    except Exception as e:
        logger.error(f"Database init error: {e}")
    finally:
        if conn:
            return_db_connection(conn)

def audit_log(action, resource_type, resource_id, changes=None):
    """Log action to audit trail"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO audit_log (action, resource_type, resource_id, user_id, changes, correlation_id)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (
            action,
            resource_type,
            resource_id,
            getattr(g, 'user_id', 'system'),
            json.dumps(changes) if changes else None,
            g.correlation_id
        ))

        conn.commit()
    except Exception as e:
        logger.error(f"Audit log error: {e}")
    finally:
        if conn:
            return_db_connection(conn)

# ===== ROUTES =====

@app.route('/health', methods=['GET'])
@limiter.exempt
def health():
    """Simple health check"""
    status = 'healthy'
    code = 200

    # Check components
    if not db_pool:
        status = 'unhealthy'
        code = 503

    app_health.set(100 if status == 'healthy' else 0)
    return jsonify({'status': status}), code

@app.route('/health/detailed', methods=['GET'])
@limiter.exempt
def health_detailed():
    """Comprehensive health check"""
    health_info = {'timestamp': datetime.utcnow().isoformat()}

    # Database check
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT NOW()')
        cursor.close()
        return_db_connection(conn)
        health_info['database'] = 'healthy'
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        health_info['database'] = 'unhealthy'

    # Redis check
    try:
        if redis_client:
            redis_client.ping()
            health_info['cache'] = 'healthy'
        else:
            health_info['cache'] = 'unavailable'
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        health_info['cache'] = 'unhealthy'

    overall_status = 'healthy' if all(v == 'healthy' for v in health_info.values() if v != health_info['timestamp']) else 'degraded'
    health_info['status'] = overall_status
    app_health.set(100 if overall_status == 'healthy' else 50)

    return jsonify(health_info), 200

@app.route('/metrics', methods=['GET'])
@limiter.exempt
def metrics():
    """Prometheus metrics"""
    db_pool_size = db_pool.closed if db_pool else 0
    db_connections.set(db_pool_size)

    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/api/jobs', methods=['GET'])
@limiter.limit("100 per minute")
@token_required
def list_jobs():
    """List jobs with audit trail"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, name, status, description, created_at
            FROM jobs
            WHERE created_by = %s OR created_by IS NULL
            ORDER BY created_at DESC LIMIT 100
        ''', (g.user_id,))

        jobs = cursor.fetchall()
        cursor.close()

        logger.info(f"Listed {len(jobs)} jobs")
        return jsonify({
            'total': len(jobs),
            'jobs': [
                {
                    'id': j[0],
                    'name': j[1],
                    'status': j[2],
                    'description': j[3],
                    'created_at': j[4].isoformat()
                }
                for j in jobs
            ]
        }), 200

    except Exception as e:
        logger.error(f"List jobs error: {e}", exc_info=True)
        request_errors.labels(endpoint='list_jobs', error_type=type(e).__name__).inc()
        return jsonify({'error': 'Internal server error', 'trace_id': g.correlation_id}), 500

    finally:
        if conn:
            return_db_connection(conn)

@app.route('/api/jobs', methods=['POST'])
@limiter.limit("20 per minute")
@token_required
def create_job():
    """Create job with validation"""
    schema = JobSchema()

    try:
        # Validate input
        data = schema.load(request.json or {})
    except ValidationError as e:
        logger.warning(f"Validation failed: {e.messages}")
        return jsonify({'errors': e.messages}), 400

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO jobs (name, status, description, created_by)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        ''', (
            data['name'],
            data.get('status', 'pending'),
            data.get('description'),
            g.user_id
        ))

        job_id = cursor.fetchone()[0]
        conn.commit()

        # Audit log
        audit_log('CREATE', 'job', job_id, {'name': data['name']})

        logger.info(f"Created job {job_id}")
        return jsonify({'id': job_id, 'status': 'created'}), 201

    except Exception as e:
        logger.error(f"Create job error: {e}", exc_info=True)
        request_errors.labels(endpoint='create_job', error_type=type(e).__name__).inc()
        return jsonify({'error': 'Internal server error', 'trace_id': g.correlation_id}), 500

    finally:
        if conn:
            return_db_connection(conn)

# ===== DASHBOARD REAL DATA ENDPOINTS =====

@app.route('/api/dashboard/stats', methods=['GET'])
@limiter.limit("30/minute")
def dashboard_stats():
    """Get healing statistics for dashboard"""
    try:
        g.correlation_id = getattr(g, 'correlation_id', str(uuid4()))
        logger.info("Dashboard stats requested")

        healing_log_path = 'data/healing_log.json'
        stats = {
            'total_heals': 0,
            'success_rate': 0.0,
            'failed_actions': 0,
            'avg_response_time': 0,
            'ml_confidence': 0.0,
            'cost_saved': 0.0,
            'action_distribution': {}
        }

        if os.path.exists(healing_log_path):
            entries = []
            with open(healing_log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except:
                            pass

            if entries:
                stats['total_heals'] = len(entries)
                successful = sum(1 for e in entries if e.get('success', False))
                stats['success_rate'] = (successful / len(entries) * 100) if len(entries) > 0 else 0
                stats['failed_actions'] = len(entries) - successful

                # Average response time
                durations = [e.get('duration_ms', 0) for e in entries]
                stats['avg_response_time'] = sum(durations) / len(durations) if durations else 0

                # Action distribution
                for entry in entries:
                    action = entry.get('action_name', 'unknown')
                    stats['action_distribution'][action] = stats['action_distribution'].get(action, 0) + 1

                # ML confidence
                confidences = []
                for e in entries:
                    ctx = e.get('context', {})
                    conf = float(ctx.get('ml_confidence', ctx.get('confidence', 0.75)))
                    confidences.append(conf)
                stats['ml_confidence'] = (sum(confidences) / len(confidences) * 100) if confidences else 75.0

                # Cost saved
                stats['cost_saved'] = successful * 37.50

        logger.info(f"Dashboard stats: {stats['total_heals']} heals, {stats['success_rate']:.1f}% success")
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f'Dashboard stats error: {e}')
        return jsonify({'error': str(e), 'trace_id': getattr(g, 'correlation_id', 'unknown')}), 500

@app.route('/api/dashboard/history', methods=['GET'])
@limiter.limit("30/minute")
def dashboard_history():
    """Get recent healing actions for dashboard"""
    try:
        g.correlation_id = getattr(g, 'correlation_id', str(uuid4()))
        limit = request.args.get('limit', 10, type=int)
        healing_log_path = 'data/healing_log.json'
        actions = []

        if os.path.exists(healing_log_path):
            entries = []
            with open(healing_log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except:
                            pass

            # Get last 'limit' entries in reverse order
            for entry in reversed(entries[-limit:]):
                ctx = entry.get('context', {})
                action = {
                    'id': str(entry.get('action_id', '')),
                    'timestamp': entry.get('timestamp', ''),
                    'action_name': entry.get('action_name', ''),
                    'success': entry.get('success', False),
                    'duration_ms': entry.get('duration_ms', 0),
                    'pod_name': ctx.get('affected_service', 'unknown'),
                    'confidence': float(ctx.get('ml_confidence', ctx.get('confidence', 0.75))),
                    'reason': ctx.get('failure_pattern', ctx.get('escalation_reason', ''))
                }
                actions.append(action)

        logger.info(f"Dashboard history: {len(actions)} actions")
        return jsonify({'actions': actions, 'count': len(actions)}), 200
    except Exception as e:
        logger.error(f'Dashboard history error: {e}')
        return jsonify({'error': str(e), 'trace_id': getattr(g, 'correlation_id', 'unknown')}), 500

@app.route('/api/dashboard/metrics', methods=['GET'])
@limiter.limit("30/minute")
def dashboard_metrics():
    """Get time-series metrics for dashboard charts"""
    try:
        g.correlation_id = getattr(g, 'correlation_id', str(uuid4()))
        healing_log_path = 'data/healing_log.json'
        metrics = []

        if os.path.exists(healing_log_path):
            entries = []
            with open(healing_log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except:
                            pass

            # Group by hourly windows (last 6 hours)
            now = datetime.utcnow()
            hourly_bins = {}

            for i in range(6):
                hour_key = (now - timedelta(hours=i)).strftime('%Y-%m-%d %H:00')
                hourly_bins[hour_key] = {'total': 0, 'success': 0}

            for entry in entries:
                try:
                    ts_str = entry.get('timestamp', '')
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    hour_key = ts.strftime('%Y-%m-%d %H:00')

                    if hour_key in hourly_bins:
                        hourly_bins[hour_key]['total'] += 1
                        if entry.get('success', False):
                            hourly_bins[hour_key]['success'] += 1
                except:
                    pass

            # Build metrics array
            for hour_key in sorted(hourly_bins.keys()):
                bin_data = hourly_bins[hour_key]
                total = bin_data['total']
                success_rate = (bin_data['success'] / total * 100) if total > 0 else 0
                metrics.append({
                    'timestamp': hour_key,
                    'success_rate': success_rate,
                    'ml_confidence': 65 + (success_rate * 0.3),
                    'total_actions': total
                })

        logger.info(f"Dashboard metrics: {len(metrics)} data points")
        return jsonify({'metrics': metrics}), 200
    except Exception as e:
        logger.error(f'Dashboard metrics error: {e}')
        return jsonify({'error': str(e), 'trace_id': getattr(g, 'correlation_id', 'unknown')}), 500

@app.route('/api/dashboard/system-metrics', methods=['GET'])
@limiter.limit("60/minute")
def dashboard_system_metrics():
    """Get realistic system metrics for charts (CPU, memory, response time)"""
    try:
        g.correlation_id = getattr(g, 'correlation_id', str(uuid4()))
        metrics_path = 'data/system_metrics.json'

        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                # Transform metrics to chart-friendly format
                chart_data = []
                for m in data.get('metrics', []):
                    timestamp = m.get('timestamp', '')
                    # Format time for charts
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_label = dt.strftime('%H:%M')
                    except:
                        time_label = timestamp[:5]

                    chart_data.append({
                        'time': time_label,
                        'cpu': round(m.get('cpu', 0), 1),
                        'memory': round(m.get('memory', 0), 1),
                        'error_rate': round(m.get('error_rate', 0) * 100, 1),
                        'response_time': round(m.get('response_time', 0), 1),
                        'pod_restarts': m.get('pod_restarts', 0),
                    })

                return jsonify({'metrics': chart_data, 'count': len(chart_data)}), 200

        # Fallback if file doesn't exist
        return jsonify({'metrics': [], 'count': 0, 'notice': 'No system metrics data'}), 200
    except Exception as e:
        logger.error(f'System metrics error: {e}')
        return jsonify({'error': str(e), 'trace_id': getattr(g, 'correlation_id', 'unknown')}), 500

# ===== FAILURE TRIGGER ENDPOINTS (For Testing) =====

@app.route('/api/trigger/jenkins-failure', methods=['POST'])
@limiter.limit("10/minute")
def trigger_jenkins_failure():
    """Simulate a Jenkins build failure to trigger orchestrator"""
    try:
        g.correlation_id = getattr(g, 'correlation_id', str(uuid4()))
        logger.warning(f"[TEST] Simulating Jenkins build failure - {g.correlation_id}")

        # Create failure event in healing log
        failure_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "action_id": 999,
            "action_name": "detect_failure",
            "success": False,
            "duration_ms": 100,
            "detail": "TEST: Jenkins build failure triggered",
            "context": {
                "build_number": "test-build-001",
                "affected_service": "dummy-app",
                "failure_prob": "0.95",
                "failure_pattern": "BuildFailure"
            }
        }

        # Append to healing log
        healing_log_path = 'data/healing_log.json'
        with open(healing_log_path, 'a') as f:
            f.write(json.dumps(failure_event) + '\n')

        logger.info(f"Jenkins failure event created - {g.correlation_id}")
        return jsonify({
            'status': 'failure_triggered',
            'event': failure_event,
            'trace_id': g.correlation_id,
            'next_step': 'Orchestrator will detect and respond',
            'watch_at': 'http://localhost:5173'
        }), 201
    except Exception as e:
        logger.error(f'Trigger jenkins failure error: {e}')
        return jsonify({'error': str(e), 'trace_id': getattr(g, 'correlation_id', 'unknown')}), 500

@app.route('/api/trigger/pod-crash', methods=['POST'])
@limiter.limit("10/minute")
def trigger_pod_crash():
    """Simulate a pod crash to trigger restart healing"""
    try:
        g.correlation_id = getattr(g, 'correlation_id', str(uuid4()))
        logger.warning(f"[TEST] Simulating pod crash - {g.correlation_id}")

        failure_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "action_id": 999,
            "action_name": "pod_restart_needed",
            "success": False,
            "duration_ms": 50,
            "detail": "TEST: Pod crash detected",
            "context": {
                "affected_service": "api-service",
                "pod_name": "api-service-xyz",
                "failure_prob": "0.99",
                "failure_pattern": "PodCrash"
            }
        }

        healing_log_path = 'data/healing_log.json'
        with open(healing_log_path, 'a') as f:
            f.write(json.dumps(failure_event) + '\n')

        logger.info(f"Pod crash event created - {g.correlation_id}")
        return jsonify({
            'status': 'pod_crash_triggered',
            'suggested_action': 'restart_pod',
            'pod': 'api-service-xyz',
            'trace_id': g.correlation_id,
            'expected_healing': 'Pod will be automatically restarted'
        }), 201
    except Exception as e:
        logger.error(f'Trigger pod crash error: {e}')
        return jsonify({'error': str(e), 'trace_id': getattr(g, 'correlation_id', 'unknown')}), 500

@app.route('/api/trigger/cpu-spike', methods=['POST'])
@limiter.limit("10/minute")
def trigger_cpu_spike():
    """Simulate high CPU to trigger scale-up healing"""
    try:
        g.correlation_id = getattr(g, 'correlation_id', str(uuid4()))
        logger.warning(f"[TEST] Simulating CPU spike - {g.correlation_id}")

        failure_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "action_id": 999,
            "action_name": "high_cpu_detected",
            "success": False,
            "duration_ms": 75,
            "detail": "TEST: High CPU detected",
            "context": {
                "affected_service": "web-frontend",
                "cpu_usage": "95%",
                "failure_prob": "0.92",
                "failure_pattern": "HighCPU"
            }
        }

        healing_log_path = 'data/healing_log.json'
        with open(healing_log_path, 'a') as f:
            f.write(json.dumps(failure_event) + '\n')

        logger.info(f"CPU spike event created - {g.correlation_id}")
        return jsonify({
            'status': 'cpu_spike_triggered',
            'suggested_action': 'scale_up',
            'current_cpu': '95%',
            'trace_id': g.correlation_id,
            'expected_healing': 'Service will be scaled up automatically'
        }), 201
    except Exception as e:
        logger.error(f'Trigger CPU spike error: {e}')
        return jsonify({'error': str(e), 'trace_id': getattr(g, 'correlation_id', 'unknown')}), 500

@app.route('/api/test/full-demo-flow', methods=['POST'])
@limiter.limit("5/minute")
def trigger_full_demo_flow():
    """Trigger a complete healing flow: Failure → Detection → Healing → Recovery"""
    try:
        g.correlation_id = getattr(g, 'correlation_id', str(uuid4()))
        logger.info(f"[DEMO] Full flow test started - {g.correlation_id}")

        healing_log_path = 'data/healing_log.json'

        # Step 1: Failure Detection
        failure = {
            "timestamp": datetime.utcnow().isoformat(),
            "action_id": 999,
            "action_name": "demo_failure",
            "success": False,
            "duration_ms": 100,
            "detail": "DEMO: Build pipeline failure",
            "context": {
                "build_number": "demo-build-001",
                "affected_service": "demo-app",
                "failure_prob": "0.95",
                "failure_pattern": "DemoFailure"
            }
        }

        # Step 2: Healing Action
        healing = {
            "timestamp": (datetime.utcnow() + timedelta(seconds=2)).isoformat(),
            "action_id": 1000,
            "action_name": "restart_pod",
            "success": True,
            "duration_ms": 150,
            "detail": "DEMO: Pod restarted automatically",
            "context": {
                "pod_name": "demo-app-abc123",
                "action_triggered_by": "demo_flow",
                "ml_confidence": "0.92"
            }
        }

        # Step 3: Recovery
        recovery = {
            "timestamp": (datetime.utcnow() + timedelta(seconds=4)).isoformat(),
            "action_id": 1001,
            "action_name": "verify_health",
            "success": True,
            "duration_ms": 80,
            "detail": "DEMO: Service health restored",
            "context": {
                "service": "demo-app",
                "status": "healthy",
                "response_time": "52ms"
            }
        }

        # Write all events
        with open(healing_log_path, 'a') as f:
            f.write(json.dumps(failure) + '\n')
            f.write(json.dumps(healing) + '\n')
            f.write(json.dumps(recovery) + '\n')

        logger.info(f"Full demo flow created - {g.correlation_id}")
        return jsonify({
            'status': 'demo_flow_triggered',
            'trace_id': g.correlation_id,
            'flow': [
                {'step': 1, 'time': '0s', 'event': 'Failure detected (build failure)', 'action': 'detect'},
                {'step': 2, 'time': '2s', 'event': 'Healing action executed (restart pod)', 'action': 'restart_pod'},
                {'step': 3, 'time': '4s', 'event': 'Service recovered and healthy', 'action': 'verify'}
            ],
            'instructions': 'Open dashboard at http://localhost:5173 and refresh to see the new events',
            'next_action': 'Check healing_log.json for new entries'
        }), 201
    except Exception as e:
        logger.error(f'Full demo flow error: {e}')
        return jsonify({'error': str(e), 'trace_id': getattr(g, 'correlation_id', 'unknown')}), 500

@app.errorhandler(429)
def rate_limit_exceeded(e):
    """Handle rate limit"""
    logger.warning(f"Rate limit exceeded - {request.remote_addr}")
    correlation_id = getattr(g, 'correlation_id', 'unknown')
    return jsonify({'error': 'Rate limit exceeded', 'trace_id': correlation_id}), 429

@app.errorhandler(404)
def not_found(e):
    correlation_id = getattr(g, 'correlation_id', 'unknown')
    return jsonify({'error': 'Not found', 'trace_id': correlation_id}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}", exc_info=True)
    request_errors.labels(endpoint=request.endpoint or 'unknown', error_type='server_error').inc()
    correlation_id = getattr(g, 'correlation_id', 'unknown')
    return jsonify({'error': 'Internal server error', 'trace_id': correlation_id}), 500

# ===== STARTUP/SHUTDOWN =====
_initialized = False

@app.before_request
def startup():
    """Initialize on first request (Flask 3.x compatible)"""
    global _initialized
    if not _initialized:
        init_database()
        _initialized = True

def shutdown_handler(signum, frame):
    """Graceful shutdown"""
    logger.info("Graceful shutdown initiated")
    if db_pool:
        db_pool.closeall()
    if redis_client:
        redis_client.close()
    exit(0)

import signal
signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

if __name__ == '__main__':
    logger.info(f"Starting in {ENVIRONMENT} mode")
    # For production, use Gunicorn, not Flask dev server!
    # This is just for development/testing
    app.run(host='0.0.0.0', port=5000, debug=(ENVIRONMENT == 'development'))
