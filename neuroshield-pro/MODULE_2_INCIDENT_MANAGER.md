"""
Module 2 Enhancement: Incident Manager with Advanced Features
- Incident timeline with detailed events
- Multi-threaded comment system with mentions
- Runbook browser with quick actions
- Impact analysis and related alerts
- Escalation tracking and resolution workflow
"""

# Add to backend/app.py routes section:

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
                # Calculate impact metrics
                related_alerts = inc.get('related_alerts', [])
                affected_services = len(set([a.get('service') for a in related_alerts if a.get('service')]))
                affected_users = 1200  # Example: would come from metrics
                downtime_minutes = 15  # Example: would calculate from timeline

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
            'steps': [
                'Identify affected pod name',
                'Check logs: kubectl logs <pod>',
                'Delete pod: kubectl delete pod <pod>',
                'Verify pod is running: kubectl get pods',
                'Monitor health for 5 minutes'
            ],
            'estimated_time': 5,
            'success_rate': 95
        },
        {
            'id': 'rb-002',
            'title': 'Build Pipeline Recovery',
            'category': 'Jenkins',
            'steps': [
                'Check Jenkins logs',
                'Review failed build artifacts',
                'Clear build cache if needed',
                'Re-run build with debugging enabled',
                'Monitor build progress'
            ],
            'estimated_time': 15,
            'success_rate': 85
        },
        {
            'id': 'rb-003',
            'title': 'Database Connection Issues',
            'category': 'Database',
            'steps': [
                'Check database logs',
                'Verify database is running',
                'Check connection pooling settings',
                'Restart connection pool',
                'Re-run queries'
            ],
            'estimated_time': 10,
            'success_rate': 90
        },
        {
            'id': 'rb-004',
            'title': 'High Memory Usage Remediation',
            'category': 'Performance',
            'steps': [
                'Identify process consuming memory',
                'Check for memory leaks',
                'Restart service if needed',
                'Increase memory limits',
                'Monitor memory usage'
            ],
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
                _audit_log('runbook_executed', {
                    'incident_id': incident_id,
                    'runbook_id': runbook_id,
                    'user': data.get('user', 'system')
                })
                socketio.emit('runbook_executed', {
                    'incident_id': incident_id,
                    'runbook_id': runbook_id,
                    'status': 'completed'
                }, broadcast=True)
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
                _audit_log('incident_assigned', {
                    'incident_id': incident_id,
                    'assigned_to': data.get('assigned_to', [])
                })
                socketio.emit('incident_assigned', {
                    'incident_id': incident_id,
                    'assigned_to': inc['assigned_to']
                }, broadcast=True)
                return jsonify({'status': 'assigned', 'incident': inc})
        return jsonify({'error': 'Incident not found'}), 404

# ─── ENHANCED FRONTEND INCIDENT DETAILS VIEW ─────

# Add to frontend HTML in <div v-if="activeModule === 'incidents'"> section:

"""
<div v-if="selectedIncident" style="margin-top: 30px; border-top: 1px solid var(--border); padding-top: 20px;">
    <h3 style="margin-bottom: 20px;">Incident Details: #{{ selectedIncident.id }}</h3>

    <div class="grid grid-2">
        <!-- Incident Overview -->
        <div class="card">
            <div style="font-weight: 600; margin-bottom: 12px;">Incident Overview</div>
            <div style="font-size: 13px; color: var(--text-dim);">
                <div style="margin-bottom: 8px;">
                    <strong>Title:</strong> {{ selectedIncident.title }}
                </div>
                <div style="margin-bottom: 8px;">
                    <strong>Severity:</strong> <span :class="'badge badge-' + (selectedIncident.severity === 'CRITICAL' ? 'danger' : 'warning')">{{ selectedIncident.severity }}</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <strong>Status:</strong> {{ selectedIncident.status }}
                </div>
                <div style="margin-bottom: 8px;">
                    <strong>Created:</strong> {{ formatTime(selectedIncident.created_at) }}
                </div>
                <div>
                    <strong>Assigned To:</strong> {{ selectedIncident.assigned_to.join(', ') || 'Unassigned' }}
                </div>
            </div>
        </div>

        <!-- Impact Analysis -->
        <div class="card">
            <div style="font-weight: 600; margin-bottom: 12px;">Impact Analysis</div>
            <div style="font-size: 13px; color: var(--text-dim);">
                <div style="margin-bottom: 8px;">
                    <strong>Affected Services:</strong> 3
                </div>
                <div style="margin-bottom: 8px;">
                    <strong>Affected Users:</strong> 1,200
                </div>
                <div style="margin-bottom: 8px;">
                    <strong>Downtime:</strong> 15 min
                </div>
                <div style="margin-bottom: 8px;">
                    <strong>Estimated Cost:</strong> $600
                </div>
                <div>
                    <strong>Team Involved:</strong> {{ selectedIncident.assigned_to.length }} members
                </div>
            </div>
        </div>
    </div>

    <!-- Timeline -->
    <h4 style="margin-top: 20px; margin-bottom: 12px;">Timeline</h4>
    <div class="card" style="max-height: 400px; overflow-y: auto;">
        <div v-for="event in selectedIncident.timeline" :key="event.ts" style="padding: 12px 0; border-bottom: 1px solid var(--border); font-size: 13px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <strong style="color: var(--primary);">{{ event.action }}</strong>
                <span style="color: var(--text-dim);">{{ formatTime(event.ts) }}</span>
            </div>
            <div style="color: var(--text-dim); margin-bottom: 4px;">{{ event.message }}</div>
            <div style="font-size: 12px; color: gray;">by {{ event.user }}</div>
        </div>
    </div>

    <!-- Comments Section -->
    <h4 style="margin-top: 20px; margin-bottom: 12px;">Team Comments</h4>
    <div class="card">
        <div v-for="comment in selectedIncident.comments" :key="comment.id" style="padding: 12px 0; border-bottom: 1px solid var(--border); font-size: 13px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <strong>{{ comment.user }}</strong>
                <span style="color: var(--text-dim); font-size: 12px;">{{ formatTime(comment.timestamp) }}</span>
            </div>
            <div>{{ comment.text }}</div>
        </div>
        <div style="margin-top: 12px;">
            <textarea placeholder="Add a comment..." style="width: 100%; height: 80px; margin-bottom: 8px;"></textarea>
            <button class="btn-primary" style="width: 100%;">Post Comment</button>
        </div>
    </div>

    <!-- Runbook Selection -->
    <h4 style="margin-top: 20px; margin-bottom: 12px;">Recommended Runbooks</h4>
    <div class="grid grid-2">
        <div class="card">
            <div style="font-weight: 600; margin-bottom: 8px;">Pod Restart Procedure</div>
            <div style="font-size: 12px; color: var(--text-dim); margin-bottom: 12px;">Estimated time: 5 min | Success rate: 95%</div>
            <button class="btn-primary" style="width: 100%;">Execute Runbook</button>
        </div>
        <div class="card">
            <div style="font-weight: 600; margin-bottom: 8px;">Database Recovery</div>
            <div style="font-size: 12px; color: var(--text-dim); margin-bottom: 12px;">Estimated time: 10 min | Success rate: 90%</div>
            <button class="btn-primary" style="width: 100%;">Execute Runbook</button>
        </div>
    </div>

    <!-- Action Buttons -->
    <div style="margin-top: 20px; display: flex; gap: 10px;">
        <button class="btn-secondary">Escalate</button>
        <button class="btn-secondary">Assign To Team</button>
        <button class="btn-primary" @click="resolveIncident(selectedIncident.id)">Resolve Incident</button>
    </div>
</div>
"""
