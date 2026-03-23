-- NeuroShield Database Initialization
-- Creates proper users, roles, and schemas with security best practices

-- Create app user with limited permissions
CREATE USER neuroshield_app WITH PASSWORD 'nA1bC2dE3fG4hI5jK6lM7nO8pQ9rS0tU1vW2xY3zA4bC5dE6fG7';

-- Create backup user
CREATE USER neuroshield_backup WITH PASSWORD 'kZ9yX8wV7uT6sR5qP4oN3mL2kJ1iH0gF9eD8cB7aZ6yX5wV4uT';

-- Create read-only user for monitoring
CREATE USER neuroshield_readonly WITH PASSWORD 'neuroshield_readonly_password';

-- Grant minimal permissions to app user
GRANT CONNECT ON DATABASE neuroshield_db TO neuroshield_app;
GRANT USAGE ON SCHEMA public TO neuroshield_app;

-- Allow app user to read/write/delete its own data only
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO neuroshield_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO neuroshield_app;

-- Grant backup user permissions
GRANT CONNECT ON DATABASE neuroshield_db TO neuroshield_backup;
ALTER ROLE neuroshield_backup SUPERUSER;  -- Only for backups

-- Grant read-only permissions
GRANT CONNECT ON DATABASE neuroshield_db TO neuroshield_readonly;
GRANT USAGE ON SCHEMA public TO neuroshield_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO neuroshield_readonly;

-- Revoke default public permissions
REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON DATABASE neuroshield_db FROM PUBLIC;

-- Enable logging of all statements
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries > 1s
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';

SELECT pg_reload_conf();

-- Create tables with audit fields
CREATE TABLE IF NOT EXISTS jobs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255) NOT NULL DEFAULT 'system',
    updated_by VARCHAR(255) DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    action VARCHAR(50) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    record_id VARCHAR(255),
    old_data JSONB,
    new_data JSONB,
    user_id VARCHAR(255) NOT NULL,
    correlation_id VARCHAR(36),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX idx_jobs_created_by ON jobs(created_by);
CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp DESC);
CREATE INDEX idx_audit_log_user ON audit_log(user_id);
CREATE INDEX idx_audit_log_table ON audit_log(table_name);

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_func() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (action, table_name, record_id, old_data, user_id, correlation_id)
        VALUES ('DELETE', TG_TABLE_NAME, OLD.id::TEXT, row_to_json(OLD), CURRENT_USER, null);
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (action, table_name, record_id, old_data, new_data, user_id, correlation_id)
        VALUES ('UPDATE', TG_TABLE_NAME, NEW.id::TEXT, row_to_json(OLD), row_to_json(NEW), CURRENT_USER, null);
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (action, table_name, record_id, new_data, user_id, correlation_id)
        VALUES ('INSERT', TG_TABLE_NAME, NEW.id::TEXT, row_to_json(NEW), CURRENT_USER, null);
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
DROP TRIGGER IF EXISTS jobs_audit_trigger ON jobs;
CREATE TRIGGER jobs_audit_trigger AFTER INSERT OR UPDATE OR DELETE ON jobs
FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();

-- Enable Row-Level Security (RLS)
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;

-- Create RLS policy - users can only see their own jobs
CREATE POLICY jobs_user_isolation ON jobs
    USING (created_by = CURRENT_USER OR created_by IS NULL)
    WITH CHECK (created_by = CURRENT_USER);

-- Grant execute on functions to app user
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO neuroshield_app;

-- Create view for app statistics
CREATE OR REPLACE VIEW job_stats AS
SELECT
    status,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_duration_seconds,
    MAX(updated_at) as last_updated
FROM jobs
GROUP BY status;

GRANT SELECT ON job_stats TO neuroshield_readonly;
GRANT SELECT ON job_stats TO neuroshield_app;

-- Connection limits
ALTER ROLE neuroshield_app WITH CONNECTION LIMIT 10;
ALTER ROLE neuroshield_readonly WITH CONNECTION LIMIT 5;

-- Ensure app user can't become superuser
ALTER ROLE neuroshield_app NOCREATEDB NOCREATEROLE;

\du+
\dp+
\dt+

COMMIT;
