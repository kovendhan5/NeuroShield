#!/bin/bash
# NeuroShield - Jenkins Pipeline Setup Script

set -e

JENKINS_URL="${JENKINS_URL:-http://localhost:8080}"
JENKINS_USER="${JENKINS_USER:-admin}"
JENKINS_PASSWORD="${JENKINS_PASSWORD:-neuroshield_pass_123}"

echo "🔧 NeuroShield Jenkins Automatic Setup"
echo "======================================"
echo ""

# Get initial admin password if exists
get_initial_password() {
    echo "📝 Retrieving Jenkins initial admin password..."
    docker exec neuroshield-jenkins cat /var/jenkins_home/secrets/initialAdminPassword 2>/dev/null || echo "Not available yet"
}

# Create the neuroshield-app-build job
create_pipeline_job() {
    echo "🔧 Creating neuroshield-app-build pipeline job..."

    JOB_XML='<?xml version="1.0" encoding="UTF-8"?>
<flow-definition plugin="workflow-job">
  <actions/>
  <description>NeuroShield Microservice Build &amp; Deploy Pipeline</description>
  <keepDependencies>false</keepDependencies>
  <properties>
    <com.cloudbees.plugins.credentials.impl.CredentialsProvider plugin="credentials">
      <domainCredentialMap>
        <java.util.TreeMap>
          <comparator class="hudson.util.CaseInsensitiveComparator"/>
        </java.util.TreeMap>
      </domainCredentialMap>
    </com.cloudbees.plugins.credentials.impl.CredentialsProvider>
    <org.jenkinsci.plugins.workflow.job.properties.PipelineTriggersJobProperty>
      <triggers>
        <com.cloudbees.plugins.bitbucket.server.trigger.RepositoryPushTrigger plugin="bitbucket-server-trigger">
          <repositoryPushAction/>
        </com.cloudbees.plugins.bitbucket.server.trigger.RepositoryPushTrigger>
      </triggers>
    </org.jenkinsci.plugins.workflow.job.properties.PipelineTriggersJobProperty>
  </properties>
  <definition class="org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition" plugin="workflow-cps">
    <script>node {
  stage("Build") {
    sh "echo NeuroShield Build Stage"
  }
  stage("Test") {
    sh "echo NeuroShield Test Stage"
  }
  stage("Deploy") {
    sh "echo NeuroShield Deploy Stage"
  }
}</script>
    <sandbox>true</sandbox>
  </definition>
  <triggers/>
</flow-definition>'

    echo "$JOB_XML" > /tmp/neuroshield-job.xml

    # Attempt to create job via Jenkins API
    echo "Attempting to create job via Jenkins API..."
    curl -s -X POST \
        "${JENKINS_URL}/createItem?name=neuroshield-app-build" \
        -H "Content-Type: application/xml" \
        -d @/tmp/neuroshield-job.xml \
        -u "${JENKINS_USER}:${JENKINS_PASSWORD}" \
        || echo "Note: Job creation may require manual setup"
}

# Configure webhook
configure_webhook() {
    echo "🔗 Configuring GitHub/GitLab webhook..."
    echo ""
    echo "Webhook URL: ${JENKINS_URL}/github-webhook/"
    echo "Or BitBucket: ${JENKINS_URL}/bitbucket-hook/"
    echo ""
    echo "Set this in your repository webhook settings"
}

# Main execution
echo ""
echo "Step 1: Initial Admin Password"
get_initial_password
echo ""

echo "Step 2: Wait for Jenkins to be fully initialized..."
sleep 5

echo ""
echo "Step 3: Creating Pipeline Job"
create_pipeline_job || echo "⚠️  Manual job creation may be required"

echo ""
echo "Step 4: Webhook Configuration"
configure_webhook

echo ""
echo "✅ Jenkins setup script completed!"
echo ""
echo "📖 Next Steps:"
echo "  1. Visit ${JENKINS_URL}"
echo "  2. Complete Jenkins setup wizard (if first time)"
echo "  3. Verify the neuroshield-app-build job exists"
echo "  4. Configure repository webhooks"
echo "  5. Enable NeuroShield healing triggers in job config"
