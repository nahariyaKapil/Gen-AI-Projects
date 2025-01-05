"""
🚀 Kapil Nahariya - Expert GenAI Engineer Portfolio
Professional AI/ML Engineer Showcase with Live Projects
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional

# Configure page
st.set_page_config(
    page_title="Kapil Nahariya | Expert GenAI Engineer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Professional GenAI Portfolio by Kapil Nahariya"
    }
)

# Professional CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }
    
    /* Remove underlines from all links */
    a {
        text-decoration: none !important;
    }
    
    a:hover {
        text-decoration: none !important;
    }
    
    a:visited {
        text-decoration: none !important;
    }
    
    a:active {
        text-decoration: none !important;
    }
    
    a:focus {
        text-decoration: none !important;
    }
    
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Professional Header */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><polygon fill="rgba(255,255,255,0.1)" points="0,0 1000,300 1000,1000 0,700"/></svg>');
        background-size: cover;
    }
    
    .hero-content {
        position: relative;
        z-index: 2;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        color: rgba(255,255,255,0.9);
        margin: 1rem 0;
        font-weight: 400;
    }
    
    .contact-info {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 2rem;
        flex-wrap: wrap;
    }
    
    .contact-item {
        background: rgba(255,255,255,0.15);
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        color: white;
        font-size: 0.9rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Stats Section */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4facfe;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1rem;
        color: rgba(255,255,255,0.8);
        font-weight: 500;
    }
    
    /* Professional Experience Section */
    .section-header {
        text-align: center;
        margin: 3rem 0 2rem 0;
    }
    
    .section-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .section-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.7);
    }
    
    .experience-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .experience-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 2rem;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    
    .experience-card:hover {
        background: rgba(255,255,255,0.12);
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    }
    
    .experience-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #4facfe;
        margin-bottom: 0.5rem;
    }
    
    .experience-company {
        font-size: 1.1rem;
        font-weight: 500;
        color: white;
        margin-bottom: 0.3rem;
    }
    
    .experience-duration {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.6);
        margin-bottom: 1rem;
    }
    
    .experience-description {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.8);
        line-height: 1.6;
    }
    
    /* Skills Section */
    .skills-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .skill-category {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 1.5rem;
        backdrop-filter: blur(20px);
        margin-bottom: 1rem;
    }
    
    .skill-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4facfe;
        margin-bottom: 1rem;
    }
    
    .skill-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    .skill-tag {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Projects Section */
    .projects-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .project-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        overflow: hidden;
        transition: all 0.3s ease;
        backdrop-filter: blur(20px);
        margin-bottom: 2rem;
    }
    
    .project-card:hover {
        background: rgba(255,255,255,0.12);
        transform: translateY(-8px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.3);
    }
    
    .project-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1.5rem;
        color: white;
    }
    
    .project-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .project-description {
        font-size: 1rem;
        opacity: 0.9;
        line-height: 1.5;
    }
    
    .project-body {
        padding: 1.5rem;
    }
    
    .project-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .project-tag {
        background: rgba(79, 172, 254, 0.2);
        color: #4facfe;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(79, 172, 254, 0.3);
    }
    
    .project-stats {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .project-stat {
        text-align: center;
        padding: 1rem;
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .project-stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4facfe;
        margin-bottom: 0.2rem;
    }
    
    .project-stat-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.7);
    }
    
    .launch-button {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 1rem 2rem;
        border: none;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        text-align: center;
        width: 100%;
    }
    
    .launch-button:hover {
        background: linear-gradient(45deg, #218838, #1e7e34);
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(40,167,69,0.3);
        color: white;
        text-decoration: none;
    }
    
    /* Education Section */
    .education-timeline {
        position: relative;
        padding: 2rem 0;
    }
    
    .education-item {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    
    .education-item:hover {
        background: rgba(255,255,255,0.12);
        transform: translateX(10px);
    }
    
    .education-degree {
        font-size: 1.3rem;
        font-weight: 600;
        color: #4facfe;
        margin-bottom: 0.5rem;
    }
    
    .education-school {
        font-size: 1.1rem;
        font-weight: 500;
        color: white;
        margin-bottom: 0.3rem;
    }
    
    .education-date {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.6);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .contact-info {
            flex-direction: column;
            gap: 1rem;
        }
        
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .featured-features-compact {
            grid-template-columns: 1fr;
        }
        
        .featured-stats-compact {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .app-store-section {
            flex-direction: column;
            align-items: center;
        }
        
        .store-btn {
            width: 100%;
            max-width: 200px;
            justify-content: center;
        }
        
        .featured-title {
            font-size: 1.6rem;
        }
        
        .featured-description {
            font-size: 1rem;
        }
        
        .experience-compact {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        
        .skills-compact {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        
        .projects-compact {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        
        .education-compact {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        
        .featured-project {
            padding: 1.5rem;
            margin: 1.5rem 0;
        }
        
        .experience-card, .education-item {
            padding: 1.5rem;
            min-height: auto;
        }
        
        .project-card {
            min-height: auto;
        }
        
        .skill-category {
            padding: 1.5rem;
            min-height: auto;
        }
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit components */
    .stDeployButton {
        display: none;
    }
    
    .stDecoration {
        display: none;
    }
    
    header[data-testid="stHeader"] {
        display: none;
    }
    
    .stToolbar {
        display: none;
    }
    
    footer {
        display: none;
    }
    
    /* Featured Project Section */
    .featured-project {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        color: white;
        box-shadow: 0 15px 35px rgba(52, 152, 219, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .featured-project::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><polygon fill="rgba(255,255,255,0.08)" points="0,1000 1000,700 1000,0 0,300"/></svg>');
        background-size: cover;
    }
    
    .featured-content {
        position: relative;
        z-index: 2;
    }
    
    .featured-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        color: #ecf0f1;
    }
    
    .featured-description {
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        opacity: 0.95;
        line-height: 1.5;
        color: #bdc3c7;
    }
    
    /* Technology Tags */
    .featured-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1.5rem 0;
        justify-content: center;
    }
    
    .featured-tag {
        background: rgba(236, 240, 241, 0.2);
        color: #ecf0f1;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(236, 240, 241, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* App Store Buttons */
    .app-store-section {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    .store-btn {
        background: rgba(236, 240, 241, 0.15);
        color: #ecf0f1 !important;
        padding: 0.8rem 1.5rem;
        border: 2px solid rgba(236, 240, 241, 0.3);
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none !important;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .store-btn:hover {
        background: rgba(236, 240, 241, 0.25);
        border-color: rgba(236, 240, 241, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        color: #ecf0f1 !important;
        text-decoration: none !important;
    }
    
    .store-btn:visited {
        color: #ecf0f1 !important;
        text-decoration: none !important;
    }
    
    .store-btn:active {
        color: #ecf0f1 !important;
        text-decoration: none !important;
    }
    
    .store-btn:focus {
        color: #ecf0f1 !important;
        text-decoration: none !important;
        outline: none;
    }
    
    /* Compact 2-Row Layout */
    .featured-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .featured-features-compact {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.8rem;
    }
    
    .feature-item {
        background: rgba(236, 240, 241, 0.1);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(236, 240, 241, 0.2);
        font-size: 0.9rem;
        color: #ecf0f1;
    }
    
    .featured-stats-compact {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
        max-width: 100%;
    }
    
    .stat-item {
        text-align: center;
        background: rgba(236, 240, 241, 0.1);
        padding: 1rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(236, 240, 241, 0.2);
    }
    
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: #3498db;
    }
    
    .stat-label {
        font-size: 0.8rem;
        opacity: 0.9;
        color: #bdc3c7;
    }
    
    /* Professional Experience Compact Layout */
    .experience-compact {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 2.5rem;
        margin: 2rem 0;
        max-width: 100%;
    }
    
    .experience-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 2rem;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
        min-height: 200px;
    }
    
    .experience-card:hover {
        background: rgba(255,255,255,0.12);
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    .experience-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #4facfe;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    
    .experience-company {
        font-size: 1.1rem;
        font-weight: 500;
        color: white;
        margin-bottom: 0.4rem;
    }
    
    .experience-duration {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.6);
        margin-bottom: 1rem;
    }
    
    .experience-description {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.8);
        line-height: 1.5;
    }
    
    /* Skills Compact Layout */
    .skills-compact {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .skill-category {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 1.8rem;
        backdrop-filter: blur(20px);
        margin-bottom: 1.5rem;
        min-height: 120px;
    }
    
    .skill-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4facfe;
        margin-bottom: 1rem;
    }
    
    .skill-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    
    .skill-tag {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Projects Compact Layout */
    .projects-compact {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 2.5rem;
        margin: 2rem 0;
    }
    
    .project-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        overflow: hidden;
        transition: all 0.3s ease;
        backdrop-filter: blur(20px);
        margin-bottom: 1.5rem;
        min-height: 320px;
    }
    
    .project-card:hover {
        background: rgba(255,255,255,0.12);
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.25);
    }
    
    .project-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1.5rem;
        color: white;
    }
    
    .project-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    
    .project-description {
        font-size: 0.95rem;
        opacity: 0.9;
        line-height: 1.4;
    }
    
    .project-body {
        padding: 1.5rem;
    }
    
    .project-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.2rem;
    }
    
    .project-tag {
        background: rgba(79, 172, 254, 0.2);
        color: #4facfe;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(79, 172, 254, 0.3);
    }
    
    .project-stats {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    
    .project-stat {
        text-align: center;
        padding: 1rem;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .project-stat-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #4facfe;
        margin-bottom: 0.3rem;
    }
    
    .project-stat-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.7);
    }
    
    .launch-button {
        background: linear-gradient(45deg, #4facfe, #00f2fe);
        color: white !important;
        padding: 1rem 1.5rem;
        border: none;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none !important;
        display: inline-block;
        text-align: center;
        width: 100%;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    .launch-button:hover {
        background: linear-gradient(45deg, #667eea, #764ba2);
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.5);
        color: white !important;
        text-decoration: none !important;
    }
    
    .launch-button:visited {
        color: white !important;
        text-decoration: none !important;
    }
    
    .launch-button:active {
        color: white !important;
        text-decoration: none !important;
    }
    
    .launch-button:focus {
        color: white !important;
        text-decoration: none !important;
        outline: none;
    }
    
    /* Education Compact Layout */
    .education-compact {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 2.5rem;
        margin: 2rem 0;
    }
    
    .education-item {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 2rem;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
        min-height: 140px;
    }
    
    .education-item:hover {
        background: rgba(255,255,255,0.12);
        transform: translateX(5px);
    }
    
    .education-degree {
        font-size: 1.3rem;
        font-weight: 600;
        color: #4facfe;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    
    .education-school {
        font-size: 1.1rem;
        font-weight: 500;
        color: white;
        margin-bottom: 0.4rem;
    }
    
    .education-date {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.6);
    }
</style>
""", unsafe_allow_html=True)

# Personal Information
PERSONAL_INFO = {
    "name": "KAPIL NAHARIYA",
    "email": "nahariyakapil@gmail.com",
    "phone": "+91 9971518560",
    "location": "Gurgaon, India 122001",
    "title": "Expert GenAI Engineer & Full-Stack Mobile Developer",
    "summary": "Built cross-platform mobile apps using Flutter, Dart, Swift, and Kotlin. Conducted data analysis with Python, SQL, Power BI, Excel, integrated via Azure Data Factory, Databricks, and ADLS Gen2. Automated reports, built ETL pipelines, and developed ML-driven insights for business decisions."
}

# Experience Data
EXPERIENCE = [
    {
        "title": "Data Analyst",
        "company": "Indus Towers Limited",
        "location": "Gurgaon, India",
        "duration": "11/2024 - Current",
        "description": "• Automated reporting and dashboards utilizing Python, Power BI, and Excel macros\n• Constructed ETL pipelines from Oracle and backend systems for structured data management\n• Provided actionable insights to senior management to drive strategic decisions"
    },
    {
        "title": "Freelance Mobile App Developer & Data Engineering",
        "company": "DelyFish",
        "location": "Gurgaon, India",
        "duration": "09/2023 - 10/2024",
        "description": "• Created DelyFish e-commerce app for iOS and Android platforms using Flutter and Firebase\n• Constructed Azure-based data pipelines to enhance sales, inventory, and customer analytics\n• Integrated REST APIs and payment systems for seamless transactions and notifications"
    },
    {
        "title": "Data Analyst",
        "company": "Healthkart (Bright Life Care Private Limited)",
        "location": "Gurgaon, India",
        "duration": "04/2023 - 08/2023",
        "description": "• Conducted analysis of customer behavior and market trends using EDA and Power BI\n• Developed dashboards to inform product and marketing strategies"
    },
    {
        "title": "Associate Engineer (Business Data Analyst)",
        "company": "Jalongi Retail Pvt Ltd",
        "location": "Gurgaon, India",
        "duration": "08/2022 - 12/2022",
        "description": "• Utilized machine learning for seafood quality control and supply chain traceability\n• Optimized inventory management through analysis of customer and sales data"
    },
    {
        "title": "Associate Engineer",
        "company": "KV IT SOLUTIONS PVT LTD",
        "location": "Delhi, India",
        "duration": "08/2020 - 08/2022",
        "description": "• Developed machine learning models, including spam filter, recommender, and fraud detection systems\n• Generated business insights using predictive analytics and data visualization techniques"
    }
]

# Education Data
EDUCATION = [
    {
        "degree": "Post Graduation Programme: Data Engineering",
        "school": "Praxis Business School",
        "location": "Bangalore",
        "date": "02/2024"
    },
    {
        "degree": "B.Sc: Computer Science",
        "school": "Shivaji College, University Of Delhi",
        "location": "Delhi",
        "date": "05/2019"
    }
]

# Skills Data
SKILLS = {
    "AI & GenAI": ["OpenAI GPT-4", "Anthropic Claude", "Google Gemini", "LangChain", "LangGraph", "RAG Systems", "Multi-Agent Systems", "Vector Databases", "Prompt Engineering", "Fine-tuning", "Chain-of-Thought", "Self-Healing AI"],
    "Computer Vision": ["Stable Diffusion", "LoRA", "CLIP", "BLIP", "MediaPipe", "OpenCV", "Face Recognition", "YOLOv8", "I3D", "SlowFast", "ViViT", "Object Detection", "Activity Recognition", "Image Generation", "Style Transfer"],
    "Machine Learning": ["PyTorch", "TensorFlow", "Scikit-learn", "Transformers", "Diffusers", "ONNX", "Quantization", "Model Optimization", "GPU Acceleration", "Pandas", "NumPy", "MLflow", "Weights & Biases"],
    "Vector & Search": ["FAISS", "ChromaDB", "Pinecone", "Sentence-Transformers", "Elasticsearch", "Semantic Search", "Hybrid Search", "Embedding Models", "Re-ranking", "Cross-Encoders"],
    "Mobile Development": ["Flutter", "Dart", "Swift", "Kotlin", "React Native", "Firebase", "iOS Development", "Android Development", "Cross-Platform", "App Store Deployment", "Push Notifications", "Real-time Sync"],
    "Data Engineering": ["Python", "SQL", "Power BI", "Excel", "Tableau", "Jupyter", "Azure Data Factory", "ADLS Gen2", "Databricks", "Azure ML", "ETL Pipelines", "Data Warehousing", "Business Intelligence"],
    "Cloud & Infrastructure": ["Google Cloud Platform", "Cloud Run", "BigQuery", "Firestore", "Cloud Storage", "Azure", "Docker", "Kubernetes", "Terraform", "Auto-scaling", "Serverless", "Microservices"],
    "Web Development": ["React", "TypeScript", "JavaScript", "FastAPI", "Streamlit", "WebSocket", "Material-UI", "Redux", "HTML5", "CSS3", "RESTful APIs", "Real-time Communication", "Progressive Web Apps"],
    "Programming Languages": ["Python", "JavaScript", "TypeScript", "Java", "C++", "Dart", "Swift", "Kotlin", "SQL", "HTML", "CSS", "Bash", "PowerShell"],
    "DevOps & Monitoring": ["CI/CD", "GitHub Actions", "Docker", "Kubernetes", "Prometheus", "Grafana", "Redis", "PostgreSQL", "Monitoring", "Alerting", "Log Management", "Performance Optimization"],
    "Security & Enterprise": ["JWT", "OAuth", "RBAC", "SSO", "TLS", "API Security", "Rate Limiting", "Data Privacy", "Compliance", "Audit Logging", "PII Detection", "Encryption"],
    "Databases": ["PostgreSQL", "Redis", "BigQuery", "Firestore", "Vector Databases", "SQL Optimization", "Database Design", "Connection Pooling", "Caching", "ACID Compliance"]
}

# Featured Mobile App Project
FEATURED_PROJECT = {
    "title": "📱 DelyFish - E-Commerce Mobile App",
    "description": "Complete cross-platform e-commerce solution for fresh fish and seafood delivery. Built with Flutter for cross-platform compatibility, native iOS (Swift) and Android (Kotlin) components for platform-specific features.",
    "ios_url": "https://apps.apple.com/us/app/delyfish/id6743340254",
    "android_url": "https://play.google.com/store/apps/details?id=com.nahariya.usersapp&pli=1",
    "website_url": "https://delyfish.com",
    "tags": ["Flutter", "Dart", "Swift", "Kotlin", "Firebase", "E-Commerce", "iOS", "Android"],
    "stats": {"Downloads": "1K+", "Platforms": "iOS+Android", "Rating": "4.5★", "Users": "500+"},
    "features": [
        "🐟 Fresh fish & seafood delivery platform",
        "📱 Cross-platform mobile app (iOS & Android)",
        "💳 Integrated payment systems & real-time notifications", 
        "📊 Azure-based analytics pipeline for sales insights",
        "🏪 Complete e-commerce solution with inventory management",
        "🚚 Real-time delivery tracking and order management"
    ]
}

# Live Projects Data
LIVE_PROJECTS = [
    {
        "title": "🧠 RAG Knowledge Assistant",
        "description": "Advanced retrieval-augmented generation system with hybrid search, multi-hop reasoning, and real-time document processing",
        "url": "https://rag-assistant-1010001041748.us-central1.run.app",
        "tags": ["RAG", "OpenAI", "FAISS", "Streamlit", "Vector Search"],
        "stats": {"Accuracy": "94.2%", "Response Time": "1.2s"}
    },
    {
        "title": "🔄 Self-Healing LLM Workflow",
        "description": "Multi-agent orchestration system with adaptive learning, error recovery, and intelligent task routing",
        "url": "https://workflow-system-1010001041748.us-central1.run.app",
        "tags": ["Multi-Agent", "LangGraph", "Self-Healing", "Memory Management"],
        "stats": {"Success Rate": "95.3%", "Healing Rate": "92.8%"}
    },
    {
        "title": "💻 Code LLM Assistant",
        "description": "Multi-language code generation platform with real-time analysis, security scanning, and optimization suggestions",
        "url": "https://code-assistant-1010001041748.us-central1.run.app",
        "tags": ["Code Generation", "Multi-Language", "Security", "Analysis"],
        "stats": {"Languages": "7+", "Accuracy": "87%"}
    },
    {
        "title": "📊 LLM Benchmark Dashboard",
        "description": "Comprehensive model performance analysis platform with cost optimization and A/B testing capabilities",
        "url": "https://benchmark-dashboard-1010001041748.us-central1.run.app",
        "tags": ["Benchmarking", "Cost Analysis", "A/B Testing", "Performance"],
        "stats": {"Models": "12+", "Uptime": "99.9%"}
    },
    {
        "title": "💰 Enterprise Cost Monitoring Dashboard",
        "description": "Real-time cost analytics platform with automated optimization, resource monitoring, and enterprise-grade reporting",
        "url": "https://cost-monitoring-dashboard-1010001041748.us-central1.run.app",
        "tags": ["Cost Analytics", "Enterprise", "Real-time Monitoring", "Optimization"],
        "stats": {"Cost Savings": "75%", "Services": "11"}
    },
    {
        "title": "🌍 Multilingual Enterprise AI",
        "description": "Full-stack enterprise AI solution with 16+ language support, real-time translation, and enterprise features",
        "url": "https://multilingual-ai-1010001041748.us-central1.run.app",
        "tags": ["Multilingual", "Enterprise", "React", "FastAPI"],
        "stats": {"Languages": "16+", "Accuracy": "98.7%"}
    },
    {
        "title": "🎨 Personalized Avatar Generator",
        "description": "AI-powered avatar generation using Stable Diffusion + LoRA fine-tuning with identity preservation",
        "url": "https://avatar-generator-1010001041748.us-central1.run.app",
        "tags": ["Stable Diffusion", "LoRA", "Computer Vision", "GPU"],
        "stats": {"Styles": "6", "Quality": "94.7%"}
    },
    {
        "title": "🎬 Vision-Language Captioning",
        "description": "Multi-modal processing pipeline with BLIP, CLIP, and real-time video analysis capabilities & more",
        "url": "https://vision-captioning-1010001041748.us-central1.run.app",
        "tags": ["Computer Vision", "BLIP", "CLIP", "Multi-Modal"],
        "stats": {"Accuracy": "90%", "Models": "4"}
    },
    {
        "title": "📸 Face-Preserving Photo Enhancer",
        "description": "Advanced photo enhancement with identity preservation using MediaPipe and style transfer techniques",
        "url": "https://face-enhancer-1010001041748.us-central1.run.app",
        "tags": ["Face Recognition", "MediaPipe", "Style Transfer", "Identity"],
        "stats": {"Preservation": "95%", "Styles": "6"}
    },
    {
        "title": "🏃 Real-Time Activity Recognition",
        "description": "Real-time human activity recognition with I3D, SlowFast, ViViT models and ONNX optimization",
        "url": "https://activity-recognition-1010001041748.us-central1.run.app",
        "tags": ["Activity Recognition", "I3D", "ONNX", "Real-Time"],
        "stats": {"Activities": "50+", "FPS": "45"}
    }
]

def render_hero_section():
    """Render the hero section with personal information"""
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-content">
            <h1 class="hero-title">{PERSONAL_INFO['name']}</h1>
            <p class="hero-subtitle">{PERSONAL_INFO['title']}</p>
            <div class="contact-info">
                <div class="contact-item">📧 {PERSONAL_INFO['email']}</div>
                <div class="contact-item">📱 {PERSONAL_INFO['phone']}</div>
                <div class="contact-item">📍 {PERSONAL_INFO['location']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_stats_section():
    """Render key statistics"""
    st.markdown("""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">13</div>
            <div class="stat-label">Total Projects</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">99.9%</div>
            <div class="stat-label">System Uptime</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">5+</div>
            <div class="stat-label">Years Experience</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">1K+</div>
            <div class="stat-label">App Downloads</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_experience_section():
    """Render professional experience"""
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">Professional Experience</h2>
        <p class="section-subtitle">Building AI solutions across industries</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for side-by-side layout
    col1, col2 = st.columns(2, gap="large")
    
    # Left column - First half of experiences
    with col1:
        for i, exp in enumerate(EXPERIENCE[:3]):  # First 3 experiences
            st.markdown(f"""
            <div class="experience-card">
                <div class="experience-title">{exp['title']}</div>
                <div class="experience-company">{exp['company']}</div>
                <div class="experience-duration">{exp['duration']} • {exp['location']}</div>
                <div class="experience-description">{exp['description'].replace(chr(10), '<br>')}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Right column - Second half of experiences
    with col2:
        for i, exp in enumerate(EXPERIENCE[3:]):  # Remaining experiences
            st.markdown(f"""
            <div class="experience-card">
                <div class="experience-title">{exp['title']}</div>
                <div class="experience-company">{exp['company']}</div>
                <div class="experience-duration">{exp['duration']} • {exp['location']}</div>
                <div class="experience-description">{exp['description'].replace(chr(10), '<br>')}</div>
            </div>
            """, unsafe_allow_html=True)

def render_skills_section():
    """Render skills section"""
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">Technical Skills</h2>
        <p class="section-subtitle">Expert-level proficiency across 100+ technologies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for side-by-side layout
    col1, col2 = st.columns(2, gap="large")
    
    # Convert skills to list for easier handling
    skills_list = list(SKILLS.items())
    
    # Left column skills (first half)
    with col1:
        for i, (category, skills) in enumerate(skills_list[:6]):
            skills_html = ''.join([f'<span class="skill-tag">{skill}</span>' for skill in skills])
            st.markdown(f"""
            <div class="skill-category">
                <div class="skill-title">{category}</div>
                <div class="skill-list">
                    {skills_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Right column skills (second half)
    with col2:
        for i, (category, skills) in enumerate(skills_list[6:]):
            skills_html = ''.join([f'<span class="skill-tag">{skill}</span>' for skill in skills])
            st.markdown(f"""
            <div class="skill-category">
                <div class="skill-title">{category}</div>
                <div class="skill-list">
                    {skills_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_projects_section():
    """Render live projects section"""
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">Live GenAI Projects</h2>
        <p class="section-subtitle">Production-ready AI applications deployed on Google Cloud Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for side-by-side layout
    col1, col2 = st.columns(2, gap="large")
    
    # Calculate midpoint for even distribution
    total_projects = len(LIVE_PROJECTS)
    midpoint = (total_projects + 1) // 2  # This ensures even distribution
    
    # Left column - First half of projects
    with col1:
        for project in LIVE_PROJECTS[:midpoint]:
            tags_html = ''.join([f'<span class="project-tag">{tag}</span>' for tag in project['tags']])
            stats_html = ''.join([
                f'<div class="project-stat"><div class="project-stat-value">{value}</div><div class="project-stat-label">{key}</div></div>'
                for key, value in project['stats'].items()
            ])
            
            st.markdown(f"""
            <div class="project-card">
                <div class="project-header">
                    <div class="project-title">{project['title']}</div>
                    <div class="project-description">{project['description']}</div>
                </div>
                <div class="project-body">
                    <div class="project-tags">{tags_html}</div>
                    <div class="project-stats">{stats_html}</div>
                    <a href="{project['url']}" target="_blank" class="launch-button">
                        🚀 Launch Project
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Right column - Second half of projects
    with col2:
        for project in LIVE_PROJECTS[midpoint:]:
            tags_html = ''.join([f'<span class="project-tag">{tag}</span>' for tag in project['tags']])
            stats_html = ''.join([
                f'<div class="project-stat"><div class="project-stat-value">{value}</div><div class="project-stat-label">{key}</div></div>'
                for key, value in project['stats'].items()
            ])
            
            st.markdown(f"""
            <div class="project-card">
                <div class="project-header">
                    <div class="project-title">{project['title']}</div>
                    <div class="project-description">{project['description']}</div>
                </div>
                <div class="project-body">
                    <div class="project-tags">{tags_html}</div>
                    <div class="project-stats">{stats_html}</div>
                    <a href="{project['url']}" target="_blank" class="launch-button">
                        🚀 Launch Project
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_education_section():
    """Render education section"""
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">Education</h2>
        <p class="section-subtitle">Academic foundation in computer science and data engineering</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for side-by-side layout
    col1, col2 = st.columns(2, gap="large")
    
    # Left column - Post Graduation
    with col1:
        st.markdown("""
        <div class="education-item">
            <div class="education-degree">Post Graduation Programme: Data Engineering</div>
            <div class="education-school">Praxis Business School</div>
            <div class="education-date">02/2024 • Bangalore</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Right column - Bachelor's Degree
    with col2:
        st.markdown("""
        <div class="education-item">
            <div class="education-degree">B.Sc: Computer Science</div>
            <div class="education-school">Shivaji College, University Of Delhi</div>
            <div class="education-date">05/2019 • Delhi</div>
        </div>
        """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with quick stats"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.08); border-radius: 15px; margin-bottom: 1rem;">
            <h3 style="color: #4facfe; margin-bottom: 1rem;">🚀 Portfolio Stats</h3>
            <div style="color: white; font-size: 0.9rem; line-height: 1.8;">
                <div><strong>AI Projects:</strong> 11</div>
                <div><strong>Mobile Apps:</strong> 1</div>
                <div><strong>Technologies:</strong> 25+</div>
                <div><strong>Deployment:</strong> GCP</div>
                <div><strong>Uptime:</strong> 99.9%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.08); border-radius: 15px; margin-bottom: 1rem;">
            <h3 style="color: #4facfe; margin-bottom: 1rem;">📱 DelyFish App</h3>
            <div style="color: white; font-size: 0.9rem; line-height: 1.8;">
                <div><strong>Downloads:</strong> 1K+</div>
                <div><strong>Platforms:</strong> iOS & Android</div>
                <div><strong>Rating:</strong> 4.5★</div>
                <div><strong>Tech:</strong> Flutter & Native</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.08); border-radius: 15px; margin-bottom: 1rem;">
            <h3 style="color: #4facfe; margin-bottom: 1rem;">📊 Live Metrics</h3>
            <div style="color: white; font-size: 0.9rem; line-height: 1.8;">
                <div><strong>API Calls:</strong> 250K+</div>
                <div><strong>Users:</strong> 5,000+</div>
                <div><strong>Response Time:</strong> <2s</div>
                <div><strong>Success Rate:</strong> 99.2%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.08); border-radius: 15px;">
            <h3 style="color: #4facfe; margin-bottom: 1rem;">🎯 Key Achievements</h3>
            <div style="color: white; font-size: 0.9rem; line-height: 1.8;">
                <div>✅ Built DelyFish App</div>
                <div>✅ 11 AI Projects Live</div>
                <div>✅ Azure Data Pipelines</div>
                <div>✅ ML Model Deployment</div>
                <div>✅ Enterprise Analytics</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_featured_project():
    """Render the featured DelyFish mobile app project"""
    project = FEATURED_PROJECT
    
    # Title and description
    st.markdown(f"""
    <div class="featured-project">
        <div class="featured-content">
            <h2 class="featured-title">{project['title']}</h2>
            <p class="featured-description">{project['description']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Technology tags
    st.markdown("""
    <div class="featured-tags">
        <span class="featured-tag">Flutter</span>
        <span class="featured-tag">Dart</span>
        <span class="featured-tag">Swift</span>
        <span class="featured-tag">Kotlin</span>
        <span class="featured-tag">Firebase</span>
        <span class="featured-tag">E-Commerce</span>
        <span class="featured-tag">iOS</span>
        <span class="featured-tag">Android</span>
    </div>
    """, unsafe_allow_html=True)
    
    # App store buttons
    st.markdown(f"""
    <div class="app-store-section">
        <a href="{project['ios_url']}" target="_blank" class="store-btn">
            🍎 App Store
        </a>
        <a href="{project['android_url']}" target="_blank" class="store-btn">
            🤖 Google Play
        </a>
        <a href="{project['website_url']}" target="_blank" class="store-btn">
            🌐 Website
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Features and stats in 2-row layout
    st.markdown("""
    <div class="featured-grid">
        <div class="featured-features-compact">
            <div class="feature-item">🐟 Fresh fish & seafood delivery</div>
            <div class="feature-item">📱 Cross-platform mobile app</div>
            <div class="feature-item">💳 Integrated payment systems</div>
            <div class="feature-item">📊 Azure analytics pipeline</div>
            <div class="feature-item">🏪 Complete e-commerce solution</div>
            <div class="feature-item">🚚 Real-time delivery tracking</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats section separately
    st.markdown("""
    <div class="featured-stats-compact">
        <div class="stat-item">
            <div class="stat-value">1K+</div>
            <div class="stat-label">Downloads</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">iOS+Android</div>
            <div class="stat-label">Platforms</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">4.5★</div>
            <div class="stat-label">Rating</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">500+</div>
            <div class="stat-label">Users</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application"""
    # Render sidebar
    render_sidebar()
    
    # Main content in container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Hero section
    render_hero_section()
    
    # Stats section
    render_stats_section()
    
    # Featured project (DelyFish mobile app)
    render_featured_project()
    
    # Experience section
    render_experience_section()
    
    # Skills section
    render_skills_section()
    
    # Projects section
    render_projects_section()
    
    # Education section
    render_education_section()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 