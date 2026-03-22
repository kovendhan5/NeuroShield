variable "resource_group_name" {
  description = "Name of Azure resource group"
  type        = string
  default     = "rg-neuroshield-prod"
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "East US"
}

variable "environment" {
  description = "Environment name (prod, staging, dev)"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "Name of AKS cluster"
  type        = string
  default     = "aks-neuroshield-prod"
}

variable "node_count" {
  description = "Initial number of nodes"
  type        = number
  default     = 1
}

variable "min_node_count" {
  description = "Minimum number of nodes in auto-scaling"
  type        = number
  default     = 1
}

variable "max_node_count" {
  description = "Maximum number of nodes in auto-scaling"
  type        = number
  default     = 3
}

variable "vm_size" {
  description = "VM size for AKS nodes"
  type        = string
  default     = "Standard_B3s"  # 4 vCores, 16GB RAM
}

variable "service_cidr" {
  description = "Service CIDR for AKS"
  type        = string
  default     = "10.1.0.0/16"
}

variable "dns_service_ip" {
  description = "DNS service IP for AKS"
  type        = string
  default     = "10.1.0.10"
}

variable "registry_name" {
  description = "Name of Azure Container Registry (lowercase, no hyphens)"
  type        = string
  default     = "acrneuroshieldprod"
}

variable "registry_sku" {
  description = "SKU for container registry"
  type        = string
  default     = "Basic"
}

variable "postgres_server_name" {
  description = "Name of PostgreSQL flexible server"
  type        = string
  default     = "psql-neuroshield-prod"
}

variable "postgres_admin_user" {
  description = "PostgreSQL admin username"
  type        = string
  default     = "dbadmin"
  sensitive   = true
}

variable "postgres_admin_password" {
  description = "PostgreSQL admin password"
  type        = string
  sensitive   = true
}

variable "postgres_database_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "neuroshield"
}

variable "redis_cache_name" {
  description = "Name of Azure Cache for Redis"
  type        = string
  default     = "redis-neuroshield-prod"
}

variable "keyvault_name" {
  description = "Name of Azure Key Vault (must be globally unique)"
  type        = string
  default     = "kv-neuroshield-prod"
}
