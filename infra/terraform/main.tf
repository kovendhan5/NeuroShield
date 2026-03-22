terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "neuroshield" {
  name     = var.resource_group_name
  location = var.location
  tags = {
    project     = "NeuroShield"
    environment = var.environment
    managed_by  = "Terraform"
  }
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "neuroshield" {
  name                = var.cluster_name
  location            = azurerm_resource_group.neuroshield.location
  resource_group_name = azurerm_resource_group.neuroshield.name
  dns_prefix          = var.cluster_name

  default_node_pool {
    name            = "nodepool1"
    node_count      = var.node_count
    vm_size         = var.vm_size
    os_disk_size_gb = 30

    enable_auto_scaling = true
    min_count           = var.min_node_count
    max_count           = var.max_node_count
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    network_policy    = "azure"
    load_balancer_sku = "standard"
    service_cidr      = var.service_cidr
    dns_service_ip    = var.dns_service_ip
  }

  addon_profile {
    http_application_routing {
      enabled = false
    }
    oms_agent {
      enabled                    = true
      log_analytics_workspace_id = azurerm_log_analytics_workspace.neuroshield.id
    }
  }

  tags = azurerm_resource_group.neuroshield.tags
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "neuroshield" {
  name                = "log-neuroshield-${var.environment}"
  location            = azurerm_resource_group.neuroshield.location
  resource_group_name = azurerm_resource_group.neuroshield.name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = azurerm_resource_group.neuroshield.tags
}

# Container Registry
resource "azurerm_container_registry" "neuroshield" {
  name                = var.registry_name
  location            = azurerm_resource_group.neuroshield.location
  resource_group_name = azurerm_resource_group.neuroshield.name
  sku                 = var.registry_sku
  admin_enabled       = true

  tags = azurerm_resource_group.neuroshield.tags
}

# PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "neuroshield" {
  name                   = var.postgres_server_name
  location               = azurerm_resource_group.neuroshield.location
  resource_group_name    = azurerm_resource_group.neuroshield.name
  administrator_login    = var.postgres_admin_user
  administrator_password = var.postgres_admin_password
  sku_name               = "B_Standard_B1ms"
  storage_mb             = 32768
  version                = "14"

  backup_retention_days = 7
  geo_redundant_backup  = false

  tags = azurerm_resource_group.neuroshield.tags
}

# PostgreSQL Database
resource "azurerm_postgresql_flexible_server_database" "neuroshield" {
  name            = var.postgres_database_name
  server_id       = azurerm_postgresql_flexible_server.neuroshield.id
  collation       = "en_US.utf8"
  charset         = "UTF8"
}

# PostgreSQL Firewall Rule (allow Azure services)
resource "azurerm_postgresql_flexible_server_firewall_rule" "azure_services" {
  name             = "allow-azure-services"
  server_id        = azurerm_postgresql_flexible_server.neuroshield.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

# Redis Cache
resource "azurerm_redis_cache" "neuroshield" {
  name                = var.redis_cache_name
  location            = azurerm_resource_group.neuroshield.location
  resource_group_name = azurerm_resource_group.neuroshield.name
  capacity            = 0  # 0 = 250MB for Basic tier
  family              = "C"
  sku_name            = "Basic"
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"

  tags = azurerm_resource_group.neuroshield.tags
}

# Key Vault
resource "azurerm_key_vault" "neuroshield" {
  name                        = var.keyvault_name
  location                    = azurerm_resource_group.neuroshield.location
  resource_group_name         = azurerm_resource_group.neuroshield.name
  enabled_for_deployment      = true
  enabled_for_disk_encryption = true
  enabled_for_template_deployment = true
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  sku_name                    = "standard"

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    key_permissions = [
      "Get",
      "List",
      "Create",
      "Delete",
    ]

    secret_permissions = [
      "Get",
      "List",
      "Set",
      "Delete",
    ]
  }

  tags = azurerm_resource_group.neuroshield.tags
}

# Key Vault Secrets
resource "azurerm_key_vault_secret" "postgres_password" {
  name         = "postgres-password"
  value        = var.postgres_admin_password
  key_vault_id = azurerm_key_vault.neuroshield.id
}

resource "azurerm_key_vault_secret" "redis_key" {
  name         = "redis-key"
  value        = azurerm_redis_cache.neuroshield.primary_access_key
  key_vault_id = azurerm_key_vault.neuroshield.id
}

resource "azurerm_key_vault_secret" "registry_password" {
  name         = "registry-password"
  value        = azurerm_container_registry.neuroshield.admin_password
  key_vault_id = azurerm_key_vault.neuroshield.id
}

# Data source for current Azure context
data "azurerm_client_config" "current" {}

# Outputs
output "aks_cluster_name" {
  value = azurerm_kubernetes_cluster.neuroshield.name
}

output "aks_cluster_id" {
  value = azurerm_kubernetes_cluster.neuroshield.id
}

output "postgres_host" {
  value = azurerm_postgresql_flexible_server.neuroshield.fqdn
}

output "redis_host" {
  value = azurerm_redis_cache.neuroshield.hostname
}

output "registry_login_server" {
  value = azurerm_container_registry.neuroshield.login_server
}

output "keyvault_id" {
  value = azurerm_key_vault.neuroshield.id
}

output "log_analytics_workspace_id" {
  value = azurerm_log_analytics_workspace.neuroshield.id
}
