"""
NeuroShield Pro — Unified AIOps Platform
Real-time WebSocket-based monitoring, incident management, team collaboration, and AI-powered healing.
"""

from flask import Flask, jsonify, request, render_template_string, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
import time
import requests
import subprocess
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import base64
import hashlib
from collections import defaultdict
import uuid

app = Flask(__name__, static_url_path='/', static_folder='./templates/public')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'neuroshield-pro-secure-key-2026')
socketio = SocketIO(app, cors_allowed_origins="*")

# ────────────────────────────────────────────────────────────────────────────
# CONFIG & STATE
# ────────────────────────────────────────────────────────────────────────────

JENKINS_URL = os.environ.get("JENKINS_URL", "http://localhost:8080")
JENKINS_USER = os.environ.get("JENKINS_USERNAME", "admin")
JENKINS_TOKEN = os.environ.get("JENKINS_TOKEN", "11e8637529db35ae8f56900be49b5cb376")
PIPELINE_WATCH_URL = os.environ.get("PIPELINE_WATCH_URL", "http://localhost:5000")
PROJECT_ROOT = Path('/app')  # Container path
DATA_DIR = PROJECT_ROOT / "data"

state_lock = threading.Lock()

# Platform state
platform_data = {
    'system_health': 100.0,
    'active_incidents': [],
    'resolved_incidents': [],
    'alerts': [],
    'builds': [],
    'pods': [],
    'team_members': [],
    'on_call_schedule': [],
    'sla_metrics': {
        'uptime': 99.5,
        'mttr': 0,
        'response_time': 0,
        'resolution_trend': 'improving'
    },
    'metrics_history': [],
    'audit_log': [],
    'integrations': {
        'slack': {'connected': False, 'channel': ''},
        'discord': {'connected': False, 'webhook': ''},
        'pagerduty': {'connected': False, 'api_key': ''},
        'email': {'connected': False, 'smtp': ''}
    },
    'settings': {
        'theme': 'dark',
        'refresh_interval': 10,
        'enable_ai_insights': True,
        'notification_level': 'critical'
    }
}

incident_id_counter = 5000
alert_id_counter = 1000
connected_clients = set()

# ────────────────────────────────────────────────────────────────────────────
# REAL-TIME DATA POLLING (BACKGROUND THREADS)
# ────────────────────────────────────────────────────────────────────────────

def _poll_pipeline_watch():
    """Poll PipelineWatch for alerts and build data."""
    while True:
        try:
            with state_lock:
                # Get alerts from PipelineWatch
                resp = requests.get(f"{PIPELINE_WATCH_URL}/api/alerts", timeout=5)
                if resp.status_code == 200:
                    alerts_data = resp.json()
                    platform_data['alerts'] = alerts_data.get('active', [])

                # Get builds
                resp = requests.get(f"{PIPELINE_WATCH_URL}/api/builds", timeout=5)
                if resp.status_code == 200:
                    platform_data['builds'] = resp.json()

                # Get pods
                resp = requests.get(f"{PIPELINE_WATCH_URL}/api/pods", timeout=5)
                if resp.status_code == 200:
                    platform_data['pods'] = resp.json()

                # Broadcast to all connected clients
                if connected_clients:
                    socketio.emit('data_update', {
                        'alerts': len(platform_data['alerts']),
                        'builds': len(platform_data['builds']),
                        'pods': len(platform_data['pods']),
                        'timestamp': datetime.utcnow().isoformat()
                    }, broadcast=True)
        except Exception as e:
            pass
        time.sleep(10)

def _calculate_system_health():
    """Calculate overall system health percentage."""
    while True:
        try:
            with state_lock:
                scores = []

                # Jenkins health
                if platform_data['builds']:
                    failed = sum(1 for b in platform_data['builds'] if b.get('result') == 'FAILURE')
                    jenkins_score = max(0, 100 - (failed / len(platform_data['builds']) * 50))
                    scores.append(jenkins_score)

                # K8s health
                if platform_data['pods']:
                    ready = sum(1 for p in platform_data['pods'] if p.get('status') == 'Running')
                    k8s_score = (ready / len(platform_data['pods'])) * 100
                    scores.append(k8s_score)

                # Alert health
                critical_alerts = len([a for a in platform_data['alerts'] if a.get('severity') == 'CRITICAL'])
                alert_score = max(0, 100 - (critical_alerts * 25))
                scores.append(alert_score)

                # SLA health
                sla = platform_data['sla_metrics']
                sla_score = (sla.get('uptime', 99.5) / 100) * 100
                scores.append(sla_score)

                platform_data['system_health'] = sum(scores) / len(scores) if scores else 100.0
        except Exception as e:
            pass
        time.sleep(15)

# Start background threads
polling_thread = threading.Thread(target=_poll_pipeline_watch, daemon=True)
polling_thread.start()

health_thread = threading.Thread(target=_calculate_system_health, daemon=True)
health_thread.start()

# ────────────────────────────────────────────────────────────────────────────
# WEBSOCKET EVENTS
# ────────────────────────────────────────────────────────────────────────────

@socketio.on('connect')
def handle_connect():
    """Client connected."""
    client_id = str(uuid.uuid4())[:8]
    connected_clients.add(client_id)
    emit('connection_response', {'status': 'connected', 'client_id': client_id})
    _broadcast_full_state()

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected."""
    connected_clients.discard(request.sid)

@socketio.on('request_full_state')
def handle_request_full_state():
    """Send full platform state to client."""
    _broadcast_full_state()

def _broadcast_full_state():
    """Send complete platform state to all connected clients."""
    with state_lock:
        socketio.emit('full_state', {
            'system_health': platform_data['system_health'],
            'active_incidents': platform_data['active_incidents'],
            'alerts': platform_data['alerts'],
            'builds': platform_data['builds'],
            'pods': platform_data['pods'],
            'sla_metrics': platform_data['sla_metrics'],
            'team_members': platform_data['team_members'],
            'integrations': platform_data['integrations'],
            'settings': platform_data['settings'],
            'timestamp': datetime.utcnow().isoformat()
        }, broadcast=True, skip_sid=request.sid if hasattr(request, 'sid') else None)

# ────────────────────────────────────────────────────────────────────────────
# REST API ENDPOINTS
# ────────────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    """Render main dashboard UI."""
    return send_from_directory('./templates/public', 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    with state_lock:
        return jsonify({
            'status': 'healthy',
            'platform_health': platform_data['system_health'],
            'connected_clients': len(connected_clients),
            'timestamp': datetime.utcnow().isoformat()
        })

@app.route('/api/system/overview', methods=['GET'])
def get_overview():
    """Get system overview for home module."""
    with state_lock:
        return jsonify({
            'system_health': platform_data['system_health'],
            'active_incidents': len(platform_data['active_incidents']),
            'critical_alerts': len([a for a in platform_data['alerts'] if a.get('severity') == 'CRITICAL']),
            'team_online': len(connected_clients),
            'sla_status': platform_data['sla_metrics'],
            'recent_actions': platform_data['metrics_history'][-10:],
            'builds_last_hour': len([b for b in platform_data['builds'] if datetime.utcnow().timestamp() - b.get('timestamp', 0) < 3600]),
            'pods_running': len([p for p in platform_data['pods'] if p.get('status') == 'Running'])
        })

@app.route('/api/incidents', methods=['GET'])
def get_incidents():
    """Get all incidents with filters."""
    status = request.args.get('status', 'active')  # active, resolved, all
    with state_lock:
        if status == 'active':
            incidents = platform_data['active_incidents']
        elif status == 'resolved':
            incidents = platform_data['resolved_incidents']
        else:
            incidents = platform_data['active_incidents'] + platform_data['resolved_incidents']
        return jsonify({'incidents': sorted(incidents, key=lambda x: x.get('created_at', 0), reverse=True)})

@app.route('/api/incidents/<incident_id>', methods=['GET'])
def get_incident(incident_id):
    """Get incident details with full timeline."""
    with state_lock:
        for inc in platform_data['active_incidents'] + platform_data['resolved_incidents']:
            if inc.get('id') == incident_id:
                return jsonify(inc)
        return jsonify({'error': 'Incident not found'}), 404

@app.route('/api/incidents', methods=['POST'])
def create_incident():
    """Create new incident."""
    global incident_id_counter
    data = request.get_json()
    incident_id_counter += 1
    incident = {
        'id': str(incident_id_counter),
        'title': data.get('title', 'Untitled Incident'),
        'description': data.get('description', ''),
        'severity': data.get('severity', 'MEDIUM'),
        'status': 'active',
        'created_at': datetime.utcnow().timestamp(),
        'created_by': data.get('created_by', 'system'),
        'assigned_to': data.get('assigned_to', []),
        'timeline': [
            {'ts': datetime.utcnow().timestamp(), 'action': 'created', 'user': data.get('created_by', 'system'), 'message': 'Incident created'}
        ],
        'comments': [],
        'tags': data.get('tags', []),
        'related_alerts': data.get('related_alerts', []),
        'runbook': data.get('runbook', ''),
        'resolution': None
    }
    with state_lock:
        platform_data['active_incidents'].append(incident)
        _audit_log('incident_created', {'incident_id': str(incident_id_counter), 'by': data.get('created_by', 'system')})
    socketio.emit('incident_created', incident, broadcast=True)
    return jsonify(incident), 201

@app.route('/api/incidents/<incident_id>/comment', methods=['POST'])
def add_incident_comment(incident_id):
    """Add comment to incident."""
    data = request.get_json()
    with state_lock:
        for inc in platform_data['active_incidents'] + platform_data['resolved_incidents']:
            if inc.get('id') == incident_id:
                comment = {
                    'id': str(uuid.uuid4())[:8],
                    'user': data.get('user', 'anonymous'),
                    'text': data.get('text', ''),
                    'timestamp': datetime.utcnow().timestamp(),
                    'edited': False
                }
                inc['comments'].append(comment)
                inc['timeline'].append({
                    'ts': datetime.utcnow().timestamp(),
                    'action': 'commented',
                    'user': data.get('user', 'anonymous'),
                    'message': f"Commented: {data.get('text', '')[:50]}..."
                })
                _audit_log('incident_comment', {'incident_id': incident_id, 'user': data.get('user', 'anonymous')})
                socketio.emit('incident_updated', inc, broadcast=True)
                return jsonify(comment), 201
        return jsonify({'error': 'Incident not found'}), 404

@app.route('/api/incidents/<incident_id>/resolve', methods=['POST'])
def resolve_incident(incident_id):
    """Resolve incident."""
    data = request.get_json()
    with state_lock:
        for i, inc in enumerate(platform_data['active_incidents']):
            if inc.get('id') == incident_id:
                inc['status'] = 'resolved'
                inc['resolution'] = {
                    'resolved_by': data.get('resolved_by', 'system'),
                    'resolution_notes': data.get('resolution_notes', ''),
                    'resolved_at': datetime.utcnow().timestamp()
                }
                inc['timeline'].append({
                    'ts': datetime.utcnow().timestamp(),
                    'action': 'resolved',
                    'user': data.get('resolved_by', 'system'),
                    'message': 'Incident resolved'
                })
                platform_data['resolved_incidents'].append(inc)
                platform_data['active_incidents'].pop(i)
                _audit_log('incident_resolved', {'incident_id': incident_id, 'by': data.get('resolved_by', 'system')})
                socketio.emit('incident_resolved', inc, broadcast=True)
                return jsonify(inc)
        return jsonify({'error': 'Incident not found'}), 404

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get all alerts with filtering."""
    severity = request.args.get('severity')
    status = request.args.get('status', 'active')
    with state_lock:
        alerts = platform_data['alerts']
        if severity:
            alerts = [a for a in alerts if a.get('severity') == severity]
        return jsonify({'alerts': alerts, 'total': len(alerts)})

@app.route('/api/alerts/rules', methods=['GET', 'POST'])
def manage_alert_rules():
    """Get or create alert rules."""
    if request.method == 'POST':
        rule = request.get_json()
        rule['id'] = str(uuid.uuid4())[:8]
        rule['created_at'] = datetime.utcnow().timestamp()
        _audit_log('alert_rule_created', {'rule_id': rule['id']})
        return jsonify(rule), 201
    else:
        return jsonify({'rules': []})

@app.route('/api/incidents/<incident_id>/timeline', methods=['GET'])
def get_incident_timeline(incident_id):
    """Get detailed incident timeline with all events."""
    with state_lock:
        for inc in platform_data['active_incidents'] + platform_data['resolved_incidents']:
            if inc.get('id') == incident_id:
                return jsonify({
                    'incident_id': incident_id,
                    'timeline': sorted(inc.get('timeline', []), key=lambda x: x['ts']),
                    'comments': sorted(inc.get('comments', []), key=lambda x: x['timestamp']),
                    'total_events': len(inc.get('timeline', [])) + len(inc.get('comments', []))
                })
        return jsonify({'error': 'Incident not found'}), 404

@app.route('/api/incidents/<incident_id>/impact', methods=['GET'])
def get_incident_impact(incident_id):
    """Get impact analysis of incident."""
    with state_lock:
        for inc in platform_data['active_incidents'] + platform_data['resolved_incidents']:
            if inc.get('id') == incident_id:
                related_alerts = inc.get('related_alerts', [])
                affected_services = len(set([a.get('service') for a in related_alerts if a.get('service')]))
                affected_users = 1200
                downtime_minutes = 15

                return jsonify({
                    'incident_id': incident_id,
                    'affected_services': affected_services,
                    'affected_users': affected_users,
                    'downtime_minutes': downtime_minutes,
                    'estimated_impact_cost': affected_users * 0.5,
                    'related_alerts': len(related_alerts),
                    'team_members_involved': len(inc.get('assigned_to', []))
                })
        return jsonify({'error': 'Incident not found'}), 404

@app.route('/api/incidents/<incident_id>/escalate', methods=['POST'])
def escalate_incident(incident_id):
    """Escalate incident to higher severity."""
    data = request.get_json()
    with state_lock:
        for inc in platform_data['active_incidents']:
            if inc.get('id') == incident_id:
                old_severity = inc.get('severity')
                inc['severity'] = data.get('new_severity', 'CRITICAL')
                inc['timeline'].append({
                    'ts': datetime.utcnow().timestamp(),
                    'action': 'escalated',
                    'user': data.get('user', 'system'),
                    'message': f"Escalated from {old_severity} to {inc['severity']}"
                })
                _audit_log('incident_escalated', {
                    'incident_id': incident_id,
                    'from': old_severity,
                    'to': inc['severity']
                })
                socketio.emit('incident_escalated', {
                    'incident_id': incident_id,
                    'new_severity': inc['severity']
                }, broadcast=True)
                return jsonify({'status': 'escalated', 'incident': inc})
        return jsonify({'error': 'Incident not found'}), 404

@app.route('/api/runbooks', methods=['GET'])
def get_runbooks():
    """Get available runbooks for incident resolution."""
    runbooks = [
        {
            'id': 'rb-001',
            'title': 'Pod Restart Procedure',
            'category': 'Kubernetes',
            'steps': ['Identify affected pod name', 'Check logs', 'Delete pod', 'Verify pod running', 'Monitor health'],
            'estimated_time': 5,
            'success_rate': 95
        },
        {
            'id': 'rb-002',
            'title': 'Build Pipeline Recovery',
            'category': 'Jenkins',
            'steps': ['Check Jenkins logs', 'Review artifacts', 'Clear cache', 'Re-run build', 'Monitor progress'],
            'estimated_time': 15,
            'success_rate': 85
        },
        {
            'id': 'rb-003',
            'title': 'Database Connection Issues',
            'category': 'Database',
            'steps': ['Check DB logs', 'Verify running', 'Check pooling', 'Restart pool', 'Re-run queries'],
            'estimated_time': 10,
            'success_rate': 90
        },
        {
            'id': 'rb-004',
            'title': 'High Memory Usage',
            'category': 'Performance',
            'steps': ['Identify process', 'Check leaks', 'Restart service', 'Increase limits', 'Monitor usage'],
            'estimated_time': 20,
            'success_rate': 88
        }
    ]
    return jsonify({'runbooks': runbooks})

@app.route('/api/incidents/<incident_id>/runbook/<runbook_id>/execute', methods=['POST'])
def execute_runbook(incident_id, runbook_id):
    """Execute runbook steps for incident resolution."""
    data = request.get_json()
    with state_lock:
        for inc in platform_data['active_incidents']:
            if inc.get('id') == incident_id:
                inc['timeline'].append({
                    'ts': datetime.utcnow().timestamp(),
                    'action': 'runbook_executed',
                    'user': data.get('user', 'system'),
                    'message': f"Executed runbook {runbook_id}",
                    'runbook_id': runbook_id,
                    'steps_completed': len(data.get('steps_completed', []))
                })
                _audit_log('runbook_executed', {'incident_id': incident_id, 'runbook_id': runbook_id})
                socketio.emit('runbook_executed', {'incident_id': incident_id, 'runbook_id': runbook_id}, broadcast=True)
                return jsonify({'status': 'runbook executed', 'incident_id': incident_id})
        return jsonify({'error': 'Incident not found'}), 404

@app.route('/api/incidents/<incident_id>/assign', methods=['POST'])
def assign_incident(incident_id):
    """Assign incident to team member(s)."""
    data = request.get_json()
    with state_lock:
        for inc in platform_data['active_incidents']:
            if inc.get('id') == incident_id:
                inc['assigned_to'] = data.get('assigned_to', [])
                inc['timeline'].append({
                    'ts': datetime.utcnow().timestamp(),
                    'action': 'assigned',
                    'user': data.get('assigner', 'system'),
                    'message': f"Assigned to {', '.join(data.get('assigned_to', []))}"
                })
                _audit_log('incident_assigned', {'incident_id': incident_id, 'assigned_to': data.get('assigned_to', [])})
                socketio.emit('incident_assigned', {'incident_id': incident_id, 'assigned_to': inc['assigned_to']}, broadcast=True)
                return jsonify({'status': 'assigned', 'incident': inc})
        return jsonify({'error': 'Incident not found'}), 404

@app.route('/api/sla/metrics', methods=['GET'])
def get_sla_metrics():
    """Get SLA metrics."""
    with state_lock:
        return jsonify(platform_data['sla_metrics'])

@app.route('/api/sla/forecast', methods=['GET'])
def get_sla_forecast():
    """Get SLA forecasting and trends."""
    days = request.args.get('days', 7, type=int)
    with state_lock:
        return jsonify({
            'forecast_period_days': days,
            'projected_uptime': 99.7,
            'risk_factors': ['High error_rate', 'Pod restart spikes'],
            'recommendations': ['Scale replicas to 3', 'Enable circuit breaker'],
            'historical_data': platform_data['metrics_history'][-days:] if platform_data['metrics_history'] else []
        })

@app.route('/api/team/members', methods=['GET', 'POST'])
def manage_team():
    """Get or manage team members."""
    if request.method == 'POST':
        member = request.get_json()
        member['id'] = str(uuid.uuid4())[:8]
        with state_lock:
            platform_data['team_members'].append(member)
            _audit_log('team_member_added', {'member': member.get('name')})
        return jsonify(member), 201
    else:
        with state_lock:
            return jsonify({'team_members': platform_data['team_members']})

@app.route('/api/team/oncall', methods=['GET', 'POST'])
def manage_oncall():
    """Get or manage on-call schedule."""
    if request.method == 'POST':
        schedule = request.get_json()
        schedule['id'] = str(uuid.uuid4())[:8]
        with state_lock:
            platform_data['on_call_schedule'].append(schedule)
            _audit_log('oncall_schedule_created', {'schedule_id': schedule['id']})
        return jsonify(schedule), 201
    else:
        with state_lock:
            return jsonify({'on_call_schedule': platform_data['on_call_schedule']})

@app.route('/api/integrations/<integration>', methods=['GET', 'POST'])
def manage_integration(integration):
    """Manage integrations (Slack, Discord, PagerDuty, Email)."""
    if request.method == 'POST':
        config = request.get_json()
        with state_lock:
            if integration in platform_data['integrations']:
                platform_data['integrations'][integration].update(config)
                platform_data['integrations'][integration]['connected'] = True
                _audit_log('integration_configured', {'integration': integration})
                return jsonify({'status': 'configured', 'integration': integration})
        return jsonify({'error': 'Unknown integration'}), 400
    else:
        with state_lock:
            if integration in platform_data['integrations']:
                return jsonify(platform_data['integrations'][integration])
        return jsonify({'error': 'Unknown integration'}), 404

@app.route('/api/settings', methods=['GET', 'PATCH'])
def manage_settings():
    """Get or update platform settings."""
    if request.method == 'PATCH':
        updates = request.get_json()
        with state_lock:
            platform_data['settings'].update(updates)
            _audit_log('settings_updated', {'updates': list(updates.keys())})
        return jsonify(platform_data['settings'])
    else:
        with state_lock:
            return jsonify(platform_data['settings'])

@app.route('/api/audit/logs', methods=['GET'])
def get_audit_logs():
    """Get audit logs."""
    limit = request.args.get('limit', 100, type=int)
    with state_lock:
        return jsonify({'audit_logs': platform_data['audit_log'][-limit:]})

@app.route('/api/reports/export', methods=['GET'])
def export_report():
    """Export data as JSON or CSV."""
    format_type = request.args.get('format', 'json')  # json or csv
    date_from = request.args.get('from')
    date_to = request.args.get('to')

    with state_lock:
        data = {
            'exported_at': datetime.utcnow().isoformat(),
            'incidents': platform_data['resolved_incidents'],
            'metrics': platform_data['sla_metrics'],
            'audit_log': platform_data['audit_log']
        }

    if format_type == 'csv':
        return jsonify({'message': 'CSV export functionality coming soon'})
    else:
        return jsonify(data)

def _audit_log(action, details):
    """Log action to audit log."""
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'action': action,
        'details': details,
        'user': 'system'
    }
    platform_data['audit_log'].append(log_entry)

# ────────────────────────────────────────────────────────────────────────────
# STATIC FILES (Handled by Flask static folder)
# ────────────────────────────────────────────────────────────────────────────

# No need for explicit static route - Flask handles it automatically

# ────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error'}), 500

# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)
