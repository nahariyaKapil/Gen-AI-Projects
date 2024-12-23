"""
Streamlit Interface for Self-Healing LLM Workflow Engine
Interactive dashboard for workflow management and monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import asyncio
from typing import Dict, List, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Self-Healing LLM Workflow Engine",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .workflow-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    .status-healthy { color: #28a745; font-weight: bold; }
    .status-degraded { color: #ffc107; font-weight: bold; }
    .status-unhealthy { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🔄 Self-Healing LLM Workflow Engine")
    st.markdown("**Automated model management with failure recovery and optimization**")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🚀 Navigation")
        page = st.selectbox(
            "Select Page",
            ["Dashboard", "Workflow Management", "System Health", "Analytics", "Settings"],
            help="Navigate between different sections"
        )
        
        st.markdown("---")
        st.markdown("### 🔧 Quick Actions")
        
        if st.button("🆕 Create Workflow", use_container_width=True):
            st.session_state.show_create_workflow = True
        
        if st.button("📊 Refresh Data", use_container_width=True):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ System Info")
        st.info("**Version:** 2.0.0")
        st.info("**Status:** 🟢 Online")
        st.info(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")

    # Main content based on selected page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Workflow Management":
        show_workflow_management()
    elif page == "System Health":
        show_system_health()
    elif page == "Analytics":
        show_analytics()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    """Main dashboard with overview metrics"""
    st.header("📊 System Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🏃‍♂️ Active Workflows",
            value=get_active_workflows_count(),
            delta="+2 from yesterday"
        )
    
    with col2:
        st.metric(
            label="✅ Success Rate",
            value="94.2%",
            delta="+1.2%"
        )
    
    with col3:
        st.metric(
            label="⚡ Avg Response Time",
            value="1.2s",
            delta="-0.3s"
        )
    
    with col4:
        st.metric(
            label="🔧 Auto-Healed",
            value="7",
            delta="+3 today"
        )
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Workflow Performance")
        show_performance_chart()
    
    with col2:
        st.subheader("🔍 Healing Actions")
        show_healing_chart()
    
    # Recent activities
    st.subheader("🕐 Recent Activities")
    show_recent_activities()

def show_workflow_management():
    """Workflow creation and management interface"""
    st.header("⚙️ Workflow Management")
    
    # Create new workflow section
    with st.expander("🆕 Create New Workflow", expanded=False):
        create_workflow_form()
    
    # Active workflows
    st.subheader("🏃‍♂️ Active Workflows")
    show_active_workflows()
    
    # Workflow history
    st.subheader("📜 Workflow History")
    show_workflow_history()

def create_workflow_form():
    """Form for creating new workflows"""
    with st.form("create_workflow"):
        st.markdown("### Configure New Workflow")
        
        col1, col2 = st.columns(2)
        
        with col1:
            workflow_name = st.text_input("Workflow Name", placeholder="My Workflow")
            workflow_type = st.selectbox(
                "Workflow Type",
                ["model_training", "inference_pipeline", "data_processing"]
            )
            priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
        
        with col2:
            model_type = st.selectbox(
                "Model Type", 
                ["GPT", "BERT", "T5", "Custom"]
            )
            max_retries = st.number_input("Max Retries", min_value=1, max_value=10, value=3)
            timeout = st.number_input("Timeout (minutes)", min_value=1, max_value=240, value=60)
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            enable_healing = st.checkbox("Enable Auto-Healing", value=True)
            enable_monitoring = st.checkbox("Enable Monitoring", value=True)
            custom_config = st.text_area(
                "Custom Configuration (JSON)",
                placeholder='{"param1": "value1", "param2": "value2"}'
            )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Create Workflow", use_container_width=True)
        
        if submitted:
            if workflow_name:
                workflow_config = {
                    "name": workflow_name,
                    "type": workflow_type,
                    "priority": priority.lower(),
                    "model_type": model_type,
                    "max_retries": max_retries,
                    "timeout": timeout * 60,
                    "enable_healing": enable_healing,
                    "enable_monitoring": enable_monitoring
                }
                
                if custom_config:
                    try:
                        custom = json.loads(custom_config)
                        workflow_config.update(custom)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON in custom configuration")
                        return
                
                # Simulate workflow creation
                with st.spinner("Creating workflow..."):
                    time.sleep(2)
                    st.success(f"✅ Workflow '{workflow_name}' created successfully!")
                    st.session_state.workflows = st.session_state.get('workflows', [])
                    st.session_state.workflows.append(workflow_config)
            else:
                st.error("Please enter a workflow name")

def show_active_workflows():
    """Display active workflows"""
    workflows = get_mock_active_workflows()
    
    for workflow in workflows:
        with st.expander(f"🏃‍♂️ {workflow['name']} ({workflow['status']})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Type:** {workflow['type']}")
                st.write(f"**Started:** {workflow['started']}")
                st.write(f"**Priority:** {workflow['priority']}")
            
            with col2:
                st.write(f"**Progress:** {workflow['progress']}%")
                st.progress(workflow['progress'] / 100)
                st.write(f"**Retries:** {workflow['retries']}")
            
            with col3:
                if st.button(f"⏸️ Pause", key=f"pause_{workflow['id']}"):
                    st.info(f"Pausing workflow {workflow['name']}")
                if st.button(f"🛑 Stop", key=f"stop_{workflow['id']}"):
                    st.warning(f"Stopping workflow {workflow['name']}")
                if st.button(f"🔍 Details", key=f"details_{workflow['id']}"):
                    show_workflow_details(workflow)

def show_system_health():
    """System health monitoring"""
    st.header("🏥 System Health Monitor")
    
    # Overall health
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Healthy")
    with col2:
        st.metric("Uptime", "99.8%", "+0.1%")
    with col3:
        st.metric("Memory Usage", "67%", "-5%")
    with col4:
        st.metric("Active Connections", "342", "+12")
    
    # Component status
    st.subheader("🔧 Component Health")
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_resource_usage()
    
    with col2:
        show_error_analysis()
    
    # Service status
    show_service_status()

def show_analytics():
    """Analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    # Time range
    col1, col2 = st.columns([1, 4])
    with col1:
        time_range = st.selectbox("Time Range", ["Last 24h", "Last 7d", "Last 30d"])
    
    # Analytics sections
    show_analytics_metrics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_workflow_trends()
    
    with col2:
        show_healing_efficiency()

def show_settings():
    """System settings"""
    st.header("⚙️ System Settings")
    
    # Configuration sections
    with st.expander("🔧 Workflow Configuration"):
        st.slider("Default Timeout (minutes)", 5, 120, 30)
        st.slider("Max Retries", 1, 10, 3)
        st.checkbox("Enable Auto-healing", value=True)
        st.checkbox("Enable Detailed Logging", value=False)
    
    with st.expander("🔔 Notifications"):
        st.checkbox("Email Notifications", value=True)
        st.checkbox("Slack Integration", value=False)
        st.text_input("Webhook URL", placeholder="https://hooks.slack.com/...")
    
    with st.expander("🏥 Health Monitoring"):
        st.slider("Health Check Interval (seconds)", 10, 300, 60)
        st.slider("Alert Threshold (%)", 50, 95, 80)
        st.checkbox("Auto-restart Failed Services", value=True)
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved successfully!")

# Helper functions
def get_active_workflows_count():
    return 5

def get_mock_active_workflows():
    return [
        {
            "id": "wf_001",
            "name": "Model Training Pipeline",
            "type": "model_training",
            "status": "Running",
            "started": "2024-01-15 10:30:00",
            "priority": "High",
            "progress": 75,
            "retries": 0
        },
        {
            "id": "wf_002",
            "name": "Data Processing Job",
            "type": "data_processing",
            "status": "Queued",
            "started": "2024-01-15 11:00:00",
            "priority": "Medium",
            "progress": 0,
            "retries": 1
        }
    ]

def show_performance_chart():
    """Performance metrics chart"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='D')
    performance = [85, 87, 89, 86, 90, 92, 88, 91, 93, 89, 94, 92, 95, 91, 96]
    
    fig = px.line(x=dates, y=performance, title="Workflow Success Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

def show_healing_chart():
    """Healing actions chart"""
    actions = ['Auto-restart', 'Memory cleanup', 'Connection reset', 'Config reload']
    counts = [12, 8, 5, 3]
    
    fig = px.bar(x=actions, y=counts, title="Healing Actions (Last 24h)")
    st.plotly_chart(fig, use_container_width=True)

def show_recent_activities():
    """Recent system activities"""
    activities = [
        {"time": "14:32", "event": "Workflow 'Data Pipeline' completed successfully", "type": "success"},
        {"time": "14:15", "event": "Auto-healed connection timeout in Service A", "type": "healing"},
        {"time": "13:58", "event": "New workflow 'Model Training' started", "type": "info"},
        {"time": "13:45", "event": "Memory usage alert cleared", "type": "warning"}
    ]
    
    for activity in activities:
        icon = {"success": "✅", "healing": "🔧", "info": "ℹ️", "warning": "⚠️"}[activity["type"]]
        st.write(f"{icon} **{activity['time']}** - {activity['event']}")

def show_workflow_details(workflow):
    """Show detailed workflow information"""
    st.info(f"Detailed view for workflow: {workflow['name']}")

def show_workflow_history():
    """Show workflow execution history"""
    history = [
        {"name": "ETL Pipeline", "status": "Completed", "duration": "45m", "ended": "13:45"},
        {"name": "Model Inference", "status": "Failed", "duration": "12m", "ended": "12:30"},
        {"name": "Data Validation", "status": "Completed", "duration": "8m", "ended": "11:15"}
    ]
    
    for item in history:
        status_color = "🟢" if item["status"] == "Completed" else "🔴"
        st.write(f"{status_color} **{item['name']}** - {item['status']} ({item['duration']}) - {item['ended']}")

def show_resource_usage():
    """Resource usage monitoring"""
    st.subheader("💻 Resource Usage")
    
    # CPU usage
    cpu_usage = 68
    st.write("**CPU Usage**")
    st.progress(cpu_usage / 100)
    st.write(f"{cpu_usage}%")
    
    # Memory usage
    memory_usage = 72
    st.write("**Memory Usage**")
    st.progress(memory_usage / 100)
    st.write(f"{memory_usage}%")
    
    # Disk usage
    disk_usage = 45
    st.write("**Disk Usage**")
    st.progress(disk_usage / 100)
    st.write(f"{disk_usage}%")

def show_error_analysis():
    """Error analysis dashboard"""
    st.subheader("🚨 Error Analysis")
    
    error_types = ['Connection Timeout', 'Memory Error', 'Config Error', 'API Error']
    error_counts = [15, 8, 3, 12]
    
    fig = px.pie(values=error_counts, names=error_types, title="Error Distribution")
    st.plotly_chart(fig, use_container_width=True)

def show_service_status():
    """Service status overview"""
    st.subheader("🔧 Service Status")
    
    services = [
        {"name": "Workflow Engine", "status": "🟢 Healthy", "uptime": "99.9%"},
        {"name": "Message Queue", "status": "🟢 Healthy", "uptime": "99.8%"},
        {"name": "Database", "status": "🟡 Warning", "uptime": "98.5%"},
        {"name": "Cache Layer", "status": "🟢 Healthy", "uptime": "99.7%"}
    ]
    
    for service in services:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{service['name']}**")
        with col2:
            st.write(service['status'])
        with col3:
            st.write(service['uptime'])

def show_analytics_metrics():
    """Analytics metrics overview"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Workflows", "1,247", "+12%")
    with col2:
        st.metric("Avg Duration", "23m", "-8%")
    with col3:
        st.metric("Success Rate", "94.2%", "+2.1%")
    with col4:
        st.metric("Cost Savings", "$2,340", "+15%")

def show_workflow_trends():
    """Workflow execution trends"""
    st.subheader("📈 Workflow Trends")
    
    dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='D')
    workflows = [12, 15, 18, 14, 20, 22, 19, 25, 28, 24, 30, 26, 32, 29, 35]
    
    fig = px.area(x=dates, y=workflows, title="Daily Workflow Count")
    st.plotly_chart(fig, use_container_width=True)

def show_healing_efficiency():
    """Healing system efficiency"""
    st.subheader("🔧 Healing Efficiency")
    
    metrics = {
        'Detection Time': '0.8s',
        'Resolution Time': '2.3s',
        'Success Rate': '96.7%',
        'Auto-resolved': '89%'
    }
    
    for metric, value in metrics.items():
        st.metric(metric, value)

if __name__ == "__main__":
    main()
