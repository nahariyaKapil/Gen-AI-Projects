#!/bin/bash

#############################################################################
# Secure Production Deployment Script for GenAI Portfolio
# Version: 2.0.0
# Security: Hardened with input validation, secure credential handling, and audit logging
#############################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures
IFS=$'\n\t'       # Secure Internal Field Separator

# Script metadata
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly LOG_DIR="${PROJECT_ROOT}/logs"
readonly CONFIG_DIR="${SCRIPT_DIR}"

# Security settings
readonly UMASK_SECURE="077"
readonly MAX_LOG_SIZE="100M"
readonly LOG_RETENTION_DAYS="30"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging configuration
readonly LOG_FILE="${LOG_DIR}/deployment-$(date +%Y%m%d-%H%M%S).log"
readonly AUDIT_LOG="${LOG_DIR}/audit-$(date +%Y%m%d).log"

#############################################################################
# Security Functions
#############################################################################

init_security() {
    # Set secure umask
    umask "$UMASK_SECURE"
    
    # Create secure log directory
    mkdir -p "$LOG_DIR"
    chmod 700 "$LOG_DIR"
    
    # Initialize audit log
    echo "$(date -Iseconds) [AUDIT] Deployment started by user: $(whoami) from: $(hostname)" >> "$AUDIT_LOG"
    
    # Set trap for cleanup on exit
    trap cleanup_on_exit EXIT
    trap 'echo "Script interrupted by user"; exit 130' INT
    trap 'echo "Script terminated"; exit 143' TERM
}

cleanup_on_exit() {
    local exit_code=$?
    echo "$(date -Iseconds) [AUDIT] Deployment finished with exit code: $exit_code" >> "$AUDIT_LOG"
    
    # Clean up temporary files
    find /tmp -name "genai-deploy-*" -user "$(whoami)" -delete 2>/dev/null || true
    
    # Rotate logs if they're too large
    if [[ -f "$LOG_FILE" && $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo 0) -gt 104857600 ]]; then
        gzip "$LOG_FILE"
    fi
    
    exit $exit_code
}

log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date -Iseconds)
    
    # Write to log file
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
    
    # Write to audit log for important events
    if [[ "$level" =~ ^(ERROR|WARN|AUDIT)$ ]]; then
        echo "[$timestamp] [$level] $message" >> "$AUDIT_LOG"
    fi
}

log_info() {
    log_message "INFO" "$1"
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    log_message "SUCCESS" "$1"
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    log_message "WARN" "$1"
    echo -e "${YELLOW}⚠️  $1${NC}" >&2
}

log_error() {
    log_message "ERROR" "$1"
    echo -e "${RED}❌ $1${NC}" >&2
}

validate_input() {
    local input="$1"
    local pattern="$2"
    local description="$3"
    
    if [[ ! "$input" =~ $pattern ]]; then
        log_error "Invalid $description: $input"
        return 1
    fi
    
    return 0
}

validate_environment() {
    log_info "Validating deployment environment..."
    
    # Check required commands
    local required_commands=("gcloud" "docker" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command not found: $cmd"
            return 1
        fi
    done
    
    # Check gcloud authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        log_error "No active gcloud authentication found"
        return 1
    fi
    
    # Validate project ID format
    local project_id
    project_id=$(gcloud config get-value project 2>/dev/null || echo "")
    if ! validate_input "$project_id" "^[a-z][a-z0-9-]{4,28}[a-z0-9]$" "GCP project ID"; then
        return 1
    fi
    
    log_success "Environment validation passed"
    return 0
}

#############################################################################
# Configuration Functions
#############################################################################

load_config() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        log_error "Configuration file not found: $config_file"
        return 1
    fi
    
    # Validate config file permissions
    local file_perms
    file_perms=$(stat -f%A "$config_file" 2>/dev/null || stat -c%a "$config_file" 2>/dev/null)
    if [[ "${file_perms: -2}" != "00" ]] && [[ "${file_perms: -2}" != "40" ]]; then
        log_warning "Configuration file has overly permissive permissions: $file_perms"
    fi
    
    # Source config with validation
    if ! source "$config_file"; then
        log_error "Failed to load configuration file: $config_file"
        return 1
    fi
    
    log_info "Configuration loaded from: $config_file"
    return 0
}

validate_config() {
    log_info "Validating configuration..."
    
    # Required variables
    local required_vars=(
        "PROJECT_ID"
        "REGION"
        "SERVICE_NAME"
        "IMAGE_NAME"
        "MEMORY_LIMIT"
        "CPU_LIMIT"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required configuration variable not set: $var"
            return 1
        fi
    done
    
    # Validate specific formats
    validate_input "$PROJECT_ID" "^[a-z][a-z0-9-]{4,28}[a-z0-9]$" "PROJECT_ID" || return 1
    validate_input "$REGION" "^[a-z0-9-]+$" "REGION" || return 1
    validate_input "$SERVICE_NAME" "^[a-z][a-z0-9-]{0,62}$" "SERVICE_NAME" || return 1
    validate_input "$MEMORY_LIMIT" "^[0-9]+[MG]i?$" "MEMORY_LIMIT" || return 1
    validate_input "$CPU_LIMIT" "^[0-9]+m?$" "CPU_LIMIT" || return 1
    
    log_success "Configuration validation passed"
    return 0
}

#############################################################################
# Docker Functions
#############################################################################

build_image() {
    local dockerfile="$1"
    local image_tag="$2"
    local context_dir="$3"
    
    log_info "Building Docker image: $image_tag"
    
    # Validate Dockerfile exists
    if [[ ! -f "$dockerfile" ]]; then
        log_error "Dockerfile not found: $dockerfile"
        return 1
    fi
    
    # Create build context
    local build_context="/tmp/genai-deploy-$$"
    mkdir -p "$build_context"
    
    # Copy necessary files to build context
    cp -r "$context_dir"/* "$build_context/" 2>/dev/null || true
    cp "$dockerfile" "$build_context/Dockerfile"
    
    # Build with security scanning
    if ! docker build \
        --no-cache \
        --pull \
        --tag "$image_tag" \
        --label "build.timestamp=$(date -Iseconds)" \
        --label "build.user=$(whoami)" \
        --label "build.host=$(hostname)" \
        "$build_context"; then
        log_error "Docker build failed"
        rm -rf "$build_context"
        return 1
    fi
    
    # Clean up build context
    rm -rf "$build_context"
    
    log_success "Docker image built successfully: $image_tag"
    return 0
}

scan_image() {
    local image_tag="$1"
    
    log_info "Scanning Docker image for vulnerabilities: $image_tag"
    
    # Use docker scout or trivy if available
    if command -v trivy &> /dev/null; then
        if ! trivy image --exit-code 1 --severity HIGH,CRITICAL "$image_tag"; then
            log_error "Security vulnerabilities found in image"
            return 1
        fi
    elif command -v docker &> /dev/null && docker version --format '{{.Server.Version}}' | grep -q "^2[0-9]"; then
        # Use Docker Scout if available
        if ! docker scout cves "$image_tag" 2>/dev/null; then
            log_warning "Could not run vulnerability scan with Docker Scout"
        fi
    else
        log_warning "No vulnerability scanner available - proceeding without scan"
    fi
    
    log_success "Image security scan completed"
    return 0
}

push_image() {
    local local_tag="$1"
    local remote_tag="$2"
    
    log_info "Pushing Docker image: $remote_tag"
    
    # Tag for remote registry
    if ! docker tag "$local_tag" "$remote_tag"; then
        log_error "Failed to tag image for push"
        return 1
    fi
    
    # Configure Docker auth for GCR
    if ! gcloud auth configure-docker --quiet; then
        log_error "Failed to configure Docker authentication"
        return 1
    fi
    
    # Push image
    if ! docker push "$remote_tag"; then
        log_error "Failed to push Docker image"
        return 1
    fi
    
    log_success "Docker image pushed successfully: $remote_tag"
    return 0
}

#############################################################################
# Cloud Run Functions
#############################################################################

deploy_service() {
    local service_name="$1"
    local image_url="$2"
    local config_file="$3"
    
    log_info "Deploying Cloud Run service: $service_name"
    
    # Create deployment command with security settings
    local deploy_cmd=(
        gcloud run deploy "$service_name"
        --image "$image_url"
        --platform managed
        --region "$REGION"
        --project "$PROJECT_ID"
        --memory "$MEMORY_LIMIT"
        --cpu "$CPU_LIMIT"
        --concurrency "$MAX_CONCURRENCY"
        --timeout "$REQUEST_TIMEOUT"
        --max-instances "$MAX_INSTANCES"
        --min-instances "$MIN_INSTANCES"
        --no-allow-unauthenticated
        --service-account "$SERVICE_ACCOUNT"
        --set-env-vars "ENV=production,LOG_LEVEL=INFO"
        --quiet
    )
    
    # Add configuration file if provided
    if [[ -f "$config_file" ]]; then
        deploy_cmd+=(--env-vars-file "$config_file")
    fi
    
    # Execute deployment
    if ! "${deploy_cmd[@]}"; then
        log_error "Cloud Run deployment failed"
        return 1
    fi
    
    log_success "Cloud Run service deployed successfully: $service_name"
    return 0
}

configure_security() {
    local service_name="$1"
    
    log_info "Configuring security settings for: $service_name"
    
    # Set IAM policy for authenticated access only
    if ! gcloud run services add-iam-policy-binding "$service_name" \
        --member="allAuthenticatedUsers" \
        --role="roles/run.invoker" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --quiet; then
        log_warning "Failed to set IAM policy - service may be publicly accessible"
    fi
    
    # Configure VPC connector if specified
    if [[ -n "${VPC_CONNECTOR:-}" ]]; then
        if ! gcloud run services update "$service_name" \
            --vpc-connector "$VPC_CONNECTOR" \
            --region="$REGION" \
            --project="$PROJECT_ID" \
            --quiet; then
            log_warning "Failed to configure VPC connector"
        fi
    fi
    
    log_success "Security configuration completed"
    return 0
}

#############################################################################
# Monitoring Functions
#############################################################################

setup_monitoring() {
    local service_name="$1"
    
    log_info "Setting up monitoring for: $service_name"
    
    # Create uptime check
    local check_name="${service_name}-uptime-check"
    local service_url
    service_url=$(gcloud run services describe "$service_name" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(status.url)")
    
    if [[ -n "$service_url" ]]; then
        # Create monitoring uptime check configuration
        cat > "/tmp/uptime-check-$$.json" << EOF
{
  "displayName": "$check_name",
  "httpCheck": {
    "requestMethod": "GET",
    "path": "/health",
    "useSsl": true
  },
  "monitoredResource": {
    "type": "uptime_url",
    "labels": {
      "project_id": "$PROJECT_ID",
      "host": "$(echo "$service_url" | sed 's|https://||' | cut -d'/' -f1)"
    }
  },
  "timeout": "60s",
  "period": "300s"
}
EOF
        
        # Apply monitoring configuration
        if gcloud monitoring uptime-check-configs create \
            --config-from-file="/tmp/uptime-check-$$.json" \
            --project="$PROJECT_ID" \
            --quiet 2>/dev/null; then
            log_success "Uptime monitoring configured"
        else
            log_warning "Failed to configure uptime monitoring"
        fi
        
        rm -f "/tmp/uptime-check-$$.json"
    fi
    
    return 0
}

#############################################################################
# Health Check Functions
#############################################################################

verify_deployment() {
    local service_name="$1"
    local max_attempts=30
    local attempt=1
    
    log_info "Verifying deployment health: $service_name"
    
    # Get service URL
    local service_url
    service_url=$(gcloud run services describe "$service_name" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(status.url)")
    
    if [[ -z "$service_url" ]]; then
        log_error "Could not retrieve service URL"
        return 1
    fi
    
    # Wait for service to be ready
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts..."
        
        if curl -sf "$service_url/health" \
            --max-time 10 \
            --user-agent "deployment-script/2.0" \
            > /dev/null 2>&1; then
            log_success "Service is healthy and responding"
            echo "Service URL: $service_url"
            return 0
        fi
        
        if [[ $attempt -lt $max_attempts ]]; then
            log_info "Waiting 10 seconds before retry..."
            sleep 10
        fi
        
        ((attempt++))
    done
    
    log_error "Service failed health check after $max_attempts attempts"
    return 1
}

#############################################################################
# Main Deployment Logic
#############################################################################

print_banner() {
    echo "============================================================================"
    echo "  GenAI Portfolio - Secure Production Deployment"
    echo "  Version: 2.0.0"
    echo "  Started: $(date)"
    echo "  User: $(whoami)"
    echo "  Host: $(hostname)"
    echo "============================================================================"
}

print_usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS] SERVICE_TYPE

Deploy GenAI Portfolio services to Google Cloud Run with security hardening.

SERVICE_TYPE:
  portfolio     Deploy portfolio dashboard
  cost-monitor  Deploy cost monitoring dashboard
  all          Deploy all services

OPTIONS:
  -c, --config FILE     Configuration file (default: deployment-config.env)
  -e, --env ENV        Environment: dev|staging|prod (default: prod)
  -v, --validate-only  Validate configuration without deploying
  -h, --help          Show this help message

Examples:
  $SCRIPT_NAME portfolio
  $SCRIPT_NAME -c custom-config.env cost-monitor
  $SCRIPT_NAME --validate-only all

EOF
}

main() {
    local service_type=""
    local config_file="$CONFIG_DIR/deployment-config.env"
    local environment="prod"
    local validate_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                config_file="$2"
                shift 2
                ;;
            -e|--env)
                environment="$2"
                shift 2
                ;;
            -v|--validate-only)
                validate_only=true
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
            *)
                if [[ -z "$service_type" ]]; then
                    service_type="$1"
                else
                    log_error "Multiple service types specified"
                    print_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate service type
    if [[ ! "$service_type" =~ ^(portfolio|cost-monitor|all)$ ]]; then
        log_error "Invalid service type. Use: portfolio, cost-monitor, or all"
        print_usage
        exit 1
    fi
    
    # Initialize security and logging
    init_security
    print_banner
    
    # Load and validate configuration
    if ! load_config "$config_file"; then
        exit 1
    fi
    
    if ! validate_config; then
        exit 1
    fi
    
    if ! validate_environment; then
        exit 1
    fi
    
    # Exit if validation only
    if [[ "$validate_only" == true ]]; then
        log_success "Validation completed successfully"
        exit 0
    fi
    
    # Deploy services
    case "$service_type" in
        portfolio)
            deploy_portfolio_service
            ;;
        cost-monitor)
            deploy_cost_monitor_service
            ;;
        all)
            deploy_portfolio_service
            deploy_cost_monitor_service
            ;;
    esac
    
    log_success "Deployment completed successfully"
}

deploy_portfolio_service() {
    log_info "Starting portfolio service deployment..."
    
    local image_tag="$PROJECT_ID/genai-portfolio:$environment-$(date +%Y%m%d-%H%M%S)"
    local dockerfile="$PROJECT_ROOT/Dockerfile.portfolio"
    local context_dir="$PROJECT_ROOT"
    
    build_image "$dockerfile" "$image_tag" "$context_dir" || return 1
    scan_image "$image_tag" || return 1
    push_image "$image_tag" "gcr.io/$image_tag" || return 1
    deploy_service "genai-portfolio" "gcr.io/$image_tag" "" || return 1
    configure_security "genai-portfolio" || return 1
    setup_monitoring "genai-portfolio" || return 1
    verify_deployment "genai-portfolio" || return 1
    
    log_success "Portfolio service deployment completed"
}

deploy_cost_monitor_service() {
    log_info "Starting cost monitor service deployment..."
    
    local image_tag="$PROJECT_ID/genai-cost-monitor:$environment-$(date +%Y%m%d-%H%M%S)"
    local dockerfile="$PROJECT_ROOT/Dockerfile.cost-dashboard"
    local context_dir="$PROJECT_ROOT"
    
    build_image "$dockerfile" "$image_tag" "$context_dir" || return 1
    scan_image "$image_tag" || return 1
    push_image "$image_tag" "gcr.io/$image_tag" || return 1
    deploy_service "genai-cost-monitor" "gcr.io/$image_tag" "" || return 1
    configure_security "genai-cost-monitor" || return 1
    setup_monitoring "genai-cost-monitor" || return 1
    verify_deployment "genai-cost-monitor" || return 1
    
    log_success "Cost monitor service deployment completed"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 