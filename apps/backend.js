const express = require('express');
const prometheus = require('prom-client');
const redis = require('redis');
const { Pool } = require('pg');
const winston = require('winston');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 5000;

// ===== LOGGER =====
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: '/app/logs/backend.log' }),
    new winston.transports.Console()
  ]
});

// ===== METRICS =====
const register = new prometheus.Registry();
prometheus.collectDefaultMetrics({ register });

const buildSuccess = new prometheus.Counter({
  name: 'pipeline_builds_success_total',
  help: 'Total successful builds',
  registers: [register]
});

const buildFailure = new prometheus.Counter({
  name: 'pipeline_builds_failure_total',
  help: 'Total failed builds',
  registers: [register]
});

const deploymentLatency = new prometheus.Histogram({
  name: 'deployment_latency_seconds',
  help: 'Deployment latency in seconds',
  buckets: [0.1, 0.5, 1, 2, 5, 10],
  registers: [register]
});

const appHealth = new prometheus.Gauge({
  name: 'app_health_percentage',
  help: 'Application health percentage',
  registers: [register]
});

const activeRequests = new prometheus.Gauge({
  name: 'http_requests_active',
  help: 'Active HTTP requests',
  registers: [register]
});

// ===== DATABASE =====
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

// ===== REDIS =====
const redisClient = redis.createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
});

redisClient.on('error', err => logger.error('Redis error:', err));
redisClient.connect();

// ===== MIDDLEWARE =====
app.use(express.json());

app.use((req, res, next) => {
  activeRequests.inc();
  res.on('finish', () => activeRequests.dec());
  next();
});

// ===== ROUTES =====

// Health check
app.get('/health', (req, res) => {
  appHealth.set(100);
  res.json({ status: 'healthy', uptime: process.uptime() });
});

// Metrics endpoint (Prometheus format)
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});

// Get pipeline status
app.get('/api/pipeline-status', async (req, res) => {
  try {
    const cacheKey = 'pipeline:status';
    const cached = await redisClient.get(cacheKey);

    if (cached) {
      return res.json(JSON.parse(cached));
    }

    const result = await pool.query(
      `SELECT id, build_number, status, duration, created_at FROM builds
       ORDER BY created_at DESC LIMIT 10`
    );

    const status = {
      total: result.rowCount,
      passing: result.rows.filter(r => r.status === 'SUCCESS').length,
      failing: result.rows.filter(r => r.status === 'FAILED').length,
      builds: result.rows
    };

    await redisClient.setEx(cacheKey, 60, JSON.stringify(status));
    res.json(status);
  } catch (error) {
    logger.error('Pipeline status error:', error);
    res.status(500).json({ error: 'Pipeline status unavailable' });
  }
});

// Record build
app.post('/api/builds', async (req, res) => {
  try {
    const { build_number, status, duration } = req.body;
    const startTime = Date.now();

    const result = await pool.query(
      `INSERT INTO builds (build_number, status, duration, created_at)
       VALUES ($1, $2, $3, NOW()) RETURNING *`,
      [build_number, status, duration]
    );

    if (status === 'SUCCESS') {
      buildSuccess.inc();
    } else if (status === 'FAILED') {
      buildFailure.inc();
    }

    deploymentLatency.observe((Date.now() - startTime) / 1000);

    // Invalidate cache
    await redisClient.del('pipeline:status');

    logger.info(`Build recorded: ${build_number} - ${status}`);
    res.status(201).json(result.rows[0]);
  } catch (error) {
    logger.error('Build recording error:', error);
    res.status(500).json({ error: 'Failed to record build' });
  }
});

// Get deployment status
app.get('/api/deployments', async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT id, build_id, status, environment, started_at, completed_at
       FROM deployments ORDER BY started_at DESC LIMIT 20`
    );

    res.json({
      total: result.rowCount,
      active: result.rows.filter(r => r.status === 'IN_PROGRESS').length,
      successful: result.rows.filter(r => r.status === 'SUCCESS').length,
      failed: result.rows.filter(r => r.status === 'FAILED').length,
      deployments: result.rows
    });
  } catch (error) {
    logger.error('Deployments error:', error);
    res.status(500).json({ error: 'Deployments unavailable' });
  }
});

// Record deployment
app.post('/api/deployments', async (req, res) => {
  try {
    const { build_id, environment, status } = req.body;

    const result = await pool.query(
      `INSERT INTO deployments (build_id, environment, status, started_at)
       VALUES ($1, $2, $3, NOW()) RETURNING *`,
      [build_id, environment, status]
    );

    logger.info(`Deployment recorded: build_id=${build_id} to ${environment}`);
    res.status(201).json(result.rows[0]);
  } catch (error) {
    logger.error('Deployment recording error:', error);
    res.status(500).json({ error: 'Failed to record deployment' });
  }
});

// Get system health
app.get('/api/health/detailed', async (req, res) => {
  try {
    const dbHealth = (await pool.query('SELECT NOW()')).rowCount > 0 ? 'healthy' : 'degraded';
    const redisHealth = (await redisClient.ping()) === 'PONG' ? 'healthy' : 'degraded';

    const result = await pool.query(
      `SELECT COUNT(*) as total,
              SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as success_count
       FROM builds WHERE created_at > NOW() - INTERVAL '1 hour'`
    );

    const health = {
      status: dbHealth === 'healthy' && redisHealth === 'healthy' ? 'healthy' : 'degraded',
      components: {
        database: dbHealth,
        cache: redisHealth
      },
      metrics: {
        builds_last_hour: parseInt(result.rows[0].total) || 0,
        success_rate: result.rows[0].total > 0
          ? ((parseInt(result.rows[0].success_count) / parseInt(result.rows[0].total)) * 100).toFixed(2)
          : 0
      }
    };

    appHealth.set(health.status === 'healthy' ? 100 : 50);
    res.json(health);
  } catch (error) {
    logger.error('Health check error:', error);
    appHealth.set(0);
    res.status(500).json({ status: 'unhealthy', error: error.message });
  }
});

// ===== INITIALIZATION =====
async function initDatabase() {
  try {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS builds (
        id SERIAL PRIMARY KEY,
        build_number INTEGER NOT NULL UNIQUE,
        status VARCHAR(50) NOT NULL,
        duration INTEGER,
        created_at TIMESTAMP DEFAULT NOW()
      );

      CREATE TABLE IF NOT EXISTS deployments (
        id SERIAL PRIMARY KEY,
        build_id INTEGER REFERENCES builds(id),
        environment VARCHAR(50) NOT NULL,
        status VARCHAR(50) NOT NULL,
        started_at TIMESTAMP DEFAULT NOW(),
        completed_at TIMESTAMP
      );

      CREATE INDEX IF NOT EXISTS idx_builds_created ON builds(created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_deployments_started ON deployments(started_at DESC);
    `);
    logger.info('Database initialized');
  } catch (error) {
    logger.error('Database initialization error:', error);
  }
}

// ===== START SERVER =====
const server = app.listen(PORT, async () => {
  await initDatabase();
  logger.info(`NeuroShield Backend App listening on port ${PORT}`);

  // Periodic health update
  setInterval(async () => {
    try {
      const health = await pool.query('SELECT NOW()');
      appHealth.set(health.rowCount > 0 ? 100 : 0);
    } catch (error) {
      appHealth.set(0);
    }
  }, 30000);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(async () => {
    await pool.end();
    await redisClient.quit();
    process.exit(0);
  });
});

module.exports = app;
