"""
Generate synthetic backlog items for evaluation.
Creates realistic software bug/feature tickets without using any client data.
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Templates for generating diverse backlog items
BUG_TEMPLATES = [
    # Authentication/Security bugs
    ("Login fails with {special_char} in password", "Users cannot authenticate when password contains {special_char}. Error message: {error_type}. Affects {impact_scope}."),
    ("Session timeout not working correctly", "User sessions remain active beyond configured timeout period of {timeout_val} minutes. Security risk as users remain authenticated. Occurs in {component}."),
    ("Password reset email not sent", "Users requesting password reset do not receive email. Email service returns {error_type}. Affects {impact_scope}."),
    ("{auth_action} fails with 401 error", "Users encounter 401 unauthorized error when attempting {auth_action}. Token validation failing in {component}. Reproducible {frequency}."),

    # API/Integration bugs
    ("API returns {status_code} for {endpoint}", "{endpoint} endpoint returns {status_code} instead of expected 200. Response body: {error_type}. Breaks {impact_scope}."),
    ("Timeout error on {api_operation}", "{api_operation} operation times out after {timeout_val} seconds. Database query performance issue. Affects {impact_scope}."),
    ("Rate limiting not enforced on {endpoint}", "{endpoint} accepts unlimited requests, causing {performance_issue}. No rate limit validation in {component}."),
    ("JSON parsing error in {endpoint}", "{endpoint} fails to parse request body. Error: {error_type}. Occurs when {condition}."),

    # Database bugs
    ("Database connection pool exhausted", "Application runs out of database connections during {load_condition}. Connection pool size: {pool_size}. Causes {performance_issue}."),
    ("Query performance degradation on {table}", "SELECT query on {table} taking {timeout_val} seconds. Missing index on {field}. Affects {impact_scope}."),
    ("Data inconsistency in {table}", "{table} contains duplicate entries for {field}. Unique constraint not enforced. Data integrity issue."),
    ("Transaction deadlock in {operation}", "{operation} encounters deadlock when {condition}. Database returns {error_type}. Affects {impact_scope}."),

    # Frontend bugs
    ("{ui_element} not responsive on {device}", "{ui_element} component breaks on {device}. Layout issue in {component}. {frequency}."),
    ("JavaScript error in {feature}", "Console shows {error_type} when using {feature}. Breaks {impact_scope}. Occurs in {browser}."),
    ("Form validation missing for {field}", "{field} input field accepts invalid data. No client-side validation. Causes {error_type} on submission."),
    ("Infinite scroll not loading more items", "Scroll event not triggering pagination. {component} stuck after {pagination_count} items. Reproducible {frequency}."),

    # Performance bugs
    ("Memory leak in {component}", "{component} exhibits memory growth over time. Reaches {memory_size}MB after {duration}. Causes {performance_issue}."),
    ("{page} load time exceeds {timeout_val} seconds", "{page} page takes {timeout_val} seconds to load during {load_condition}. Performance bottleneck in {component}."),
    ("High CPU usage during {operation}", "{operation} causes CPU spike to {cpu_percent}%. Inefficient algorithm in {component}. Affects {impact_scope}."),

    # Integration/External bugs
    ("Third-party {service} integration failing", "Integration with {service} returns {error_type}. API key validation issue. Affects {impact_scope}."),
    ("Webhook delivery failures", "Webhooks not delivered to {endpoint}. Retry mechanism failing. {frequency}."),
]

FEATURE_TEMPLATES = [
    ("Add {feature_action} capability to {component}", "Enable users to {feature_action} within {component}. Improves {benefit}. Priority: {priority}."),
    ("Implement {security_feature} for {component}", "Add {security_feature} to enhance security of {component}. Prevents {security_risk}."),
    ("Support {data_format} export from {feature}", "Allow users to export data from {feature} in {data_format} format. Requested by {impact_scope}."),
    ("Add filtering by {field} in {feature}", "Implement filter option for {field} in {feature}. Improves {benefit}."),
    ("Create {report_type} dashboard", "Build dashboard showing {report_type} metrics. Include {metric_count} visualizations. Helps {impact_scope}."),
    ("Integrate with {external_service}", "Add integration with {external_service} for {integration_purpose}. Enables {benefit}."),
    ("Bulk {operation} functionality", "Allow users to perform {operation} on multiple items simultaneously. Improves {benefit}."),
    ("Add {notification_type} notifications", "Implement {notification_type} notifications for {event_type} events. Configurable frequency."),
]

TASK_TEMPLATES = [
    ("Upgrade {library} to version {version}", "Update {library} dependency from current version to {version}. Includes security patches and {benefit}."),
    ("Refactor {component} for better {quality_attr}", "Restructure {component} codebase to improve {quality_attr}. Reduce technical debt."),
    ("Add unit tests for {component}", "Increase test coverage for {component}. Current coverage: {coverage}%. Target: {target_coverage}%."),
    ("Update documentation for {feature}", "Revise user documentation for {feature}. Add examples and troubleshooting guide."),
    ("Optimize {database_operation} queries", "Improve performance of {database_operation} by adding indexes and query optimization. Current: {timeout_val}s, Target: <{target_time}s."),
    ("Setup monitoring for {metric}", "Configure alerting for {metric} threshold breaches. Integrate with monitoring dashboard."),
]

# Value pools for template substitution
SPECIAL_CHARS = ["@", "#", "$", "%", "&", "special characters", "unicode characters"]
ERROR_TYPES = ["NullPointerException", "timeout error", "connection refused", "invalid JSON", "unauthorized",
               "validation error", "constraint violation", "internal server error"]
COMPONENTS = ["Authentication", "API Gateway", "Database", "Frontend", "Backend", "User Service",
              "Payment Service", "Notification Service", "Reporting Module", "Admin Panel"]
IMPACT_SCOPES = ["all users", "admin users", "premium tier users", "mobile users", "10% of users", "specific customers"]
FREQUENCIES = ["consistently", "intermittently", "occasionally", "during peak hours", "in production only"]
STATUS_CODES = ["500", "503", "404", "401", "403", "400", "502"]
ENDPOINTS = ["/api/users", "/api/orders", "/api/products", "/api/auth", "/api/reports", "/api/payments"]
TIMEOUT_VALS = ["30", "60", "10", "5", "120"]
PERFORMANCE_ISSUES = ["system slowdown", "service unavailability", "cascading failures", "degraded performance"]
CONDITIONS = ["using special characters", "with large datasets", "during concurrent requests", "on first request"]
TABLES = ["users", "orders", "products", "transactions", "audit_logs", "sessions"]
FIELDS = ["user_id", "email", "order_date", "product_id", "timestamp", "status"]
POOL_SIZES = ["50", "100", "20", "10"]
UI_ELEMENTS = ["Navigation menu", "Search bar", "Data table", "Modal dialog", "Dropdown", "Form"]
DEVICES = ["mobile devices", "tablet", "iPhone", "Android", "small screens"]
BROWSERS = ["Chrome", "Firefox", "Safari", "Edge"]
PAGES = ["Dashboard", "Report", "User Profile", "Settings", "Admin Panel"]
MEMORY_SIZES = ["500", "1000", "2000", "5000"]
DURATIONS = ["1 hour", "4 hours", "8 hours", "24 hours"]
CPU_PERCENTS = ["80", "90", "95", "100"]
SERVICES = ["payment gateway", "email service", "SMS provider", "analytics platform"]
LOAD_CONDITIONS = ["peak traffic", "high load", "batch processing", "concurrent users"]
PAGINATION_COUNTS = ["20", "50", "100"]

FEATURE_ACTIONS = ["export", "import", "duplicate", "archive", "share", "schedule"]
SECURITY_FEATURES = ["two-factor authentication", "role-based access control", "audit logging", "encryption"]
SECURITY_RISKS = ["unauthorized access", "data leakage", "privilege escalation"]
DATA_FORMATS = ["CSV", "JSON", "PDF", "Excel"]
REPORT_TYPES = ["sales", "user activity", "performance", "error", "audit"]
METRIC_COUNTS = ["5", "10", "15"]
EXTERNAL_SERVICES = ["Salesforce", "Slack", "Google Analytics", "Stripe"]
INTEGRATION_PURPOSES = ["data synchronization", "notifications", "payment processing", "analytics"]
BENEFITS = ["user productivity", "workflow efficiency", "user experience", "data visibility"]
NOTIFICATION_TYPES = ["email", "in-app", "SMS", "push"]
EVENT_TYPES = ["order completion", "user registration", "payment success", "error occurrence"]

LIBRARIES = ["React", "Django", "PostgreSQL driver", "Redis client", "logging library"]
VERSIONS = ["3.0.0", "2.5.1", "4.1.0", "1.8.0"]
QUALITY_ATTRS = ["maintainability", "performance", "testability", "readability"]
COVERAGES = ["45", "60", "55", "40"]
TARGET_COVERAGES = ["80", "85", "90"]
DATABASE_OPERATIONS = ["user lookup", "order retrieval", "report generation", "data aggregation"]
TARGET_TIMES = ["1", "2", "0.5"]
METRICS = ["response time", "error rate", "CPU usage", "memory usage", "disk space"]

PRIORITIES = ["Critical", "High", "Medium", "Low"]
STATUSES = ["Open", "In Progress", "Resolved", "Closed", "Deferred"]
TAGS_POOL = [
    ["authentication", "security"],
    ["api", "integration"],
    ["database", "performance"],
    ["frontend", "ui"],
    ["backend", "service"],
    ["bug", "critical"],
    ["feature-request"],
    ["technical-debt"],
    ["security", "compliance"],
    ["performance", "optimization"]
]

def substitute_template(template, item_type):
    """Fill in template with random values"""
    title, description = template

    # Create substitution map
    substitutions = {
        'special_char': random.choice(SPECIAL_CHARS),
        'error_type': random.choice(ERROR_TYPES),
        'component': random.choice(COMPONENTS),
        'impact_scope': random.choice(IMPACT_SCOPES),
        'timeout_val': random.choice(TIMEOUT_VALS),
        'auth_action': random.choice(["login", "logout", "registration", "password reset"]),
        'frequency': random.choice(FREQUENCIES),
        'status_code': random.choice(STATUS_CODES),
        'endpoint': random.choice(ENDPOINTS),
        'api_operation': random.choice(["user creation", "order processing", "data retrieval", "file upload"]),
        'performance_issue': random.choice(PERFORMANCE_ISSUES),
        'condition': random.choice(CONDITIONS),
        'table': random.choice(TABLES),
        'field': random.choice(FIELDS),
        'pool_size': random.choice(POOL_SIZES),
        'ui_element': random.choice(UI_ELEMENTS),
        'device': random.choice(DEVICES),
        'browser': random.choice(BROWSERS),
        'page': random.choice(PAGES),
        'memory_size': random.choice(MEMORY_SIZES),
        'duration': random.choice(DURATIONS),
        'cpu_percent': random.choice(CPU_PERCENTS),
        'service': random.choice(SERVICES),
        'load_condition': random.choice(LOAD_CONDITIONS),
        'pagination_count': random.choice(PAGINATION_COUNTS),
        'feature_action': random.choice(FEATURE_ACTIONS),
        'security_feature': random.choice(SECURITY_FEATURES),
        'security_risk': random.choice(SECURITY_RISKS),
        'data_format': random.choice(DATA_FORMATS),
        'feature': random.choice(["Dashboard", "Reports", "User Management", "Settings"]),
        'report_type': random.choice(REPORT_TYPES),
        'metric_count': random.choice(METRIC_COUNTS),
        'external_service': random.choice(EXTERNAL_SERVICES),
        'integration_purpose': random.choice(INTEGRATION_PURPOSES),
        'benefit': random.choice(BENEFITS),
        'notification_type': random.choice(NOTIFICATION_TYPES),
        'event_type': random.choice(EVENT_TYPES),
        'library': random.choice(LIBRARIES),
        'version': random.choice(VERSIONS),
        'quality_attr': random.choice(QUALITY_ATTRS),
        'coverage': random.choice(COVERAGES),
        'target_coverage': random.choice(TARGET_COVERAGES),
        'database_operation': random.choice(DATABASE_OPERATIONS),
        'target_time': random.choice(TARGET_TIMES),
        'metric': random.choice(METRICS),
        'operation': random.choice(["delete", "update", "approve", "export"]),
        'priority': random.choice(PRIORITIES),
    }

    # Substitute all placeholders
    filled_title = title.format(**{k: v for k, v in substitutions.items() if '{' + k + '}' in title})
    filled_desc = description.format(**{k: v for k, v in substitutions.items() if '{' + k + '}' in description})

    return filled_title, filled_desc

def generate_backlog_items(n_bugs=400, n_features=250, n_tasks=150):
    """Generate synthetic backlog items"""
    items = []
    item_id = 1

    # Generate bugs
    for i in range(n_bugs):
        template = random.choice(BUG_TEMPLATES)
        title, description = substitute_template(template, "Bug")

        items.append({
            "ID": f"BUG-{item_id:04d}",
            "Title": title,
            "Description": description,
            "Type": "Bug",
            "Priority": random.choice(PRIORITIES),
            "Status": random.choice(STATUSES),
            "Component": random.choice(COMPONENTS),
            "Tags": ", ".join(random.choice(TAGS_POOL))
        })
        item_id += 1

    # Generate features
    for i in range(n_features):
        template = random.choice(FEATURE_TEMPLATES)
        title, description = substitute_template(template, "Feature")

        items.append({
            "ID": f"FEAT-{item_id:04d}",
            "Title": title,
            "Description": description,
            "Type": "Feature",
            "Priority": random.choice(PRIORITIES),
            "Status": random.choice(STATUSES),
            "Component": random.choice(COMPONENTS),
            "Tags": ", ".join(random.choice(TAGS_POOL))
        })
        item_id += 1

    # Generate tasks
    for i in range(n_tasks):
        template = random.choice(TASK_TEMPLATES)
        title, description = substitute_template(template, "Task")

        items.append({
            "ID": f"TASK-{item_id:04d}",
            "Title": title,
            "Description": description,
            "Type": "Task",
            "Priority": random.choice(PRIORITIES),
            "Status": random.choice(STATUSES),
            "Component": random.choice(COMPONENTS),
            "Tags": ", ".join(random.choice(TAGS_POOL))
        })
        item_id += 1

    return pd.DataFrame(items)

def main():
    """Generate and save synthetic backlog data"""
    print("Generating synthetic backlog items...")

    # Generate items
    df = generate_backlog_items(n_bugs=400, n_features=250, n_tasks=150)

    # Save to CSV
    output_path = Path("evaluation/synthetic_data")
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "synthetic_backlog.csv"
    df.to_csv(output_file, index=False)

    print(f"\n[SUCCESS] Generated {len(df)} synthetic backlog items")
    print(f"   - Bugs: {len(df[df['Type'] == 'Bug'])}")
    print(f"   - Features: {len(df[df['Type'] == 'Feature'])}")
    print(f"   - Tasks: {len(df[df['Type'] == 'Task'])}")
    print(f"\n   Saved to: {output_file}")

    # Print sample
    print(f"\nSample items:")
    print(df[['ID', 'Title', 'Type']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
