"""
RAG Knowledge Assistant - Streamlit Interface
Interactive dashboard for document processing and Q&A
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Any, Optional

# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .document-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    .status-healthy { color: #28a745; font-weight: bold; }
    .status-processing { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .result-card {
        background: white;
        color: #333;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 2px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 RAG Knowledge Assistant</h1>
        <p>Advanced Document Processing & Intelligent Q&A System</p>
        <p><strong>Upload • Process • Query • Discover</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🚀 Navigation")
        page = st.selectbox(
            "Select Page",
            ["Document Upload", "Query Interface", "Document Library", "Analytics", "System Health"],
            help="Navigate between different sections"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", get_document_count(), "+2")
        with col2:
            st.metric("Queries", get_query_count(), "+15")
        
        st.markdown("---")
        st.markdown("### ℹ️ System Info")
        st.info("**Status:** 🟢 Online")
        st.info("**Version:** 2.0.0")
        st.info(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")

    # Main content based on selected page
    if page == "Document Upload":
        show_document_upload()
    elif page == "Query Interface":
        show_query_interface()
    elif page == "Document Library":
        show_document_library()
    elif page == "Analytics":
        show_analytics()
    elif page == "System Health":
        show_system_health()

def show_document_upload():
    """Document upload and processing interface"""
    st.header("📁 Document Upload & Processing")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>1,247</h3>
            <p>Documents Processed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>98.7%</h3>
            <p>Processing Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>2.3s</h3>
            <p>Avg Processing Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>847</h3>
            <p>Total Chunks</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'txt', 'docx', 'md', 'csv', 'json'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX, MD, CSV, JSON"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size} bytes)")
            
            # Processing options
            with st.expander("⚙️ Processing Options"):
                col_a, col_b = st.columns(2)
                with col_a:
                    chunk_size = st.slider("Chunk Size", 100, 1000, 500)
                    overlap = st.slider("Overlap", 0, 200, 50)
                with col_b:
                    enable_ocr = st.checkbox("Enable OCR", value=True)
                    enable_nlp = st.checkbox("Enable NLP Processing", value=True)
            
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                process_documents(uploaded_files, chunk_size, overlap, enable_ocr, enable_nlp)
    
    with col2:
        st.subheader("📋 Processing Queue")
        show_processing_queue()
        
        st.subheader("✅ Recent Uploads")
        show_recent_uploads()

def show_query_interface():
    """Query interface for asking questions"""
    st.header("🤔 Intelligent Query Interface")
    
    # Query performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", "2,847", "+127")
    with col2:
        st.metric("Avg Response Time", "0.8s", "-0.2s")
    with col3:
        st.metric("Relevance Score", "94.2%", "+2.1%")
    with col4:
        st.metric("User Satisfaction", "97.8%", "+1.5%")
    
    st.markdown("---")
    
    # Query interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Ask Your Questions")
        
        query = st.text_area(
            "What would you like to know?",
            height=120,
            placeholder="Ask questions about your uploaded documents...\nFor example: 'What are the main findings in the research paper?' or 'Summarize the key points from the financial report.'"
        )
        
        # Query options
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            top_k = st.selectbox("Results to show", [3, 5, 10, 15], index=1)
        with col_b:
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.1)
        with col_c:
            query_mode = st.selectbox("Query Mode", ["Semantic", "Keyword", "Hybrid"])
        
        if st.button("🔍 Search", type="primary", use_container_width=True) and query:
            perform_query(query, top_k, similarity_threshold, query_mode)
    
    with col2:
        st.subheader("🎯 Query Suggestions")
        suggestions = [
            "What are the main topics discussed?",
            "Summarize the key findings",
            "What are the conclusions?",
            "List the important dates mentioned",
            "What recommendations are made?"
        ]
        
        for i, suggestion in enumerate(suggestions):
            if st.button(f"💡 {suggestion}", key=f"suggestion_{i}"):
                st.session_state.query_input = suggestion
                st.rerun()
        
        st.subheader("📈 Query History")
        show_query_history()

def show_document_library():
    """Document library and management"""
    st.header("📚 Document Library")
    
    # Search and filter
    col1, col2, col3 = st.columns(3)
    with col1:
        search_term = st.text_input("🔍 Search documents", placeholder="Search by name, content, or tags...")
    with col2:
        filter_type = st.selectbox("Filter by type", ["All", "PDF", "TXT", "DOCX", "MD", "CSV"])
    with col3:
        sort_by = st.selectbox("Sort by", ["Date Added", "Name", "Size", "Relevance"])
    
    # Document list
    documents = get_mock_documents()
    
    for doc in documents:
        with st.expander(f"📄 {doc['name']} ({doc['type']})"):
            col_info, col_actions = st.columns([3, 1])
            
            with col_info:
                st.write(f"**Size:** {doc['size']}")
                st.write(f"**Added:** {doc['date_added']}")
                st.write(f"**Chunks:** {doc['chunks']}")
                st.write(f"**Description:** {doc['description']}")
            
            with col_actions:
                if st.button("🔍 Query", key=f"query_{doc['id']}"):
                    st.session_state.selected_doc = doc['id']
                if st.button("📊 Analyze", key=f"analyze_{doc['id']}"):
                    show_document_analysis(doc)
                if st.button("🗑️ Delete", key=f"delete_{doc['id']}"):
                    delete_document(doc['id'])

def show_analytics():
    """Analytics and insights dashboard"""
    st.header("📊 Analytics Dashboard")
    
    # Time range selector
    col1, col2 = st.columns([1, 4])
    with col1:
        time_range = st.selectbox("Time Range", ["Last 7 days", "Last 30 days", "Last 90 days"])
    
    # Analytics charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Query Volume")
        show_query_volume_chart()
    
    with col2:
        st.subheader("📊 Document Types")
        show_document_types_chart()
    
    # Additional metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Response Times")
        show_response_times_chart()
    
    with col2:
        st.subheader("🎯 Popular Queries")
        show_popular_queries()

def show_system_health():
    """System health and monitoring"""
    st.header("🏥 System Health Monitor")
    
    # Overall health status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🟢 Healthy</h3>
            <p>Overall Status</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Uptime", "99.8%", "+0.1%")
    
    with col3:
        st.metric("Memory Usage", "67%", "-5%")
    
    # Component health
    st.subheader("🔧 Component Status")
    
    components = [
        {"name": "Document Processor", "status": "🟢 Healthy", "details": "Processing normally"},
        {"name": "Retrieval Engine", "status": "🟢 Healthy", "details": "Vector search operational"},
        {"name": "Database", "status": "🟢 Healthy", "details": "All connections active"},
        {"name": "API Gateway", "status": "🟡 Warning", "details": "High response times"}
    ]
    
    for component in components:
        with st.expander(f"{component['status']} {component['name']}"):
            st.write(component['details'])
            if "Warning" in component['status']:
                st.warning("Requires attention")

# Helper functions
def get_document_count():
    return 23

def get_query_count():
    return 156

def process_documents(files, chunk_size, overlap, enable_ocr, enable_nlp):
    """Process uploaded documents"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        progress = (i + 1) / len(files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {file.name}...")
        time.sleep(1)  # Simulate processing
    
    status_text.text("✅ All documents processed successfully!")
    st.success(f"Processed {len(files)} documents with {chunk_size} chunk size")

def perform_query(query, top_k, threshold, mode):
    """Perform document query"""
    with st.spinner("Searching documents..."):
        time.sleep(2)  # Simulate search
        
        st.success("Search completed!")
        
        # Mock results
        st.subheader("📄 Search Results")
        for i in range(min(top_k, 3)):
            with st.container():
                st.markdown(f"""
                <div class="result-card">
                    <h4>Result {i+1} (Relevance: {0.95 - i*0.1:.2f})</h4>
                    <p>This is a sample result from your documents. In a real implementation, 
                    this would contain the relevant text chunks that match your query about "{query}".</p>
                    <p><strong>Source:</strong> Document_{i+1}.pdf | <strong>Page:</strong> {i+2}</p>
                </div>
                """, unsafe_allow_html=True)

def show_processing_queue():
    """Show document processing queue"""
    queue_items = [
        {"name": "Report_2024.pdf", "status": "Processing"},
        {"name": "Analysis.docx", "status": "Queued"},
        {"name": "Data.csv", "status": "Queued"}
    ]
    
    for item in queue_items:
        status_icon = "⚡" if item["status"] == "Processing" else "⏳"
        st.write(f"{status_icon} {item['name']}")

def show_recent_uploads():
    """Show recent uploads"""
    recent = [
        {"name": "Research.pdf", "time": "2 min ago"},
        {"name": "Summary.txt", "time": "15 min ago"},
        {"name": "Notes.md", "time": "1 hour ago"}
    ]
    
    for item in recent:
        st.write(f"✅ {item['name']} ({item['time']})")

def show_query_history():
    """Show recent queries"""
    history = [
        "What are the main findings?",
        "Summarize key points",
        "List recommendations"
    ]
    
    for i, query in enumerate(history):
        if st.button(f"🔄 {query[:20]}...", key=f"history_{i}"):
            st.session_state.query_input = query

def get_mock_documents():
    """Get mock document data"""
    return [
        {
            "id": 1,
            "name": "Research_Report_2024.pdf",
            "type": "PDF",
            "size": "2.4 MB",
            "date_added": "2024-01-15",
            "chunks": 45,
            "description": "Annual research findings and analysis"
        },
        {
            "id": 2,
            "name": "Meeting_Notes.txt",
            "type": "TXT",
            "size": "156 KB",
            "date_added": "2024-01-14",
            "chunks": 12,
            "description": "Weekly team meeting notes and action items"
        }
    ]

def show_document_analysis(doc):
    """Show document analysis"""
    st.info(f"Analysis for {doc['name']} would appear here in a real implementation")

def delete_document(doc_id):
    """Delete document"""
    st.success(f"Document {doc_id} deleted successfully")

def show_query_volume_chart():
    """Show query volume chart"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='D')
    volumes = [20, 25, 30, 28, 35, 40, 42, 38, 45, 50, 48, 52, 55, 58, 60]
    
    fig = px.line(x=dates, y=volumes, title="Daily Query Volume")
    st.plotly_chart(fig, use_container_width=True)

def show_document_types_chart():
    """Show document types chart"""
    types = ['PDF', 'TXT', 'DOCX', 'MD', 'CSV']
    counts = [45, 23, 18, 12, 8]
    
    fig = px.pie(values=counts, names=types, title="Document Types")
    st.plotly_chart(fig, use_container_width=True)

def show_response_times_chart():
    """Show response times chart"""
    times = [0.8, 0.9, 0.7, 1.1, 0.6, 0.8, 0.9, 0.7, 0.8, 0.6]
    fig = px.histogram(x=times, title="Response Time Distribution", nbins=10)
    st.plotly_chart(fig, use_container_width=True)

def show_popular_queries():
    """Show popular queries"""
    queries = [
        "What are the main findings?",
        "Summarize the document",
        "List key recommendations",
        "What are the conclusions?",
        "Show important dates"
    ]
    
    for i, query in enumerate(queries):
        st.write(f"{i+1}. {query}")

if __name__ == "__main__":
    main() 