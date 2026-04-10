variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (staging / production)"
  type        = string
  default     = "staging"
}

variable "app_image" {
  description = "Docker image for loci-api (ECR URI or public)"
  type        = string
}

variable "qdrant_cloud_url" {
  description = "Qdrant Cloud cluster URL"
  type        = string
  sensitive   = true
}

variable "qdrant_api_key" {
  description = "Qdrant Cloud API key"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Aurora Serverless master password"
  type        = string
  sensitive   = true
}

variable "cors_origins" {
  description = "Comma-separated allowed CORS origins"
  type        = string
  default     = ""
}

variable "fargate_cpu" {
  description = "Fargate task CPU units"
  type        = number
  default     = 512
}

variable "fargate_memory" {
  description = "Fargate task memory (MB)"
  type        = number
  default     = 1024
}

variable "min_capacity" {
  description = "Minimum number of Fargate tasks"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum number of Fargate tasks"
  type        = number
  default     = 10
}
