import streamlit as st
import os
from PIL import Image
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime
import json
import requests
import pandas as pd 
import numpy as np 
from streamlit_lottie import st_lottie
from dilitracer_integration import DILITracerAnalyzer

# Page config
st.set_page_config(
    page_title="Organoid Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #764ba2;
        transform: translateY(-2px);
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Header with animation
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 class='main-header'>Organoid Analysis Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #6B7280;'>Powered by DILITracer AI</p>", unsafe_allow_html=True)
    
# Workflow selection
st.markdown("---")
workflow_tab1, workflow_tab2 = st.tabs(["🧬 DILITracer (Toxicity Analysis)", "🔬 Morphology Analysis"])

with workflow_tab1:
    # Load animation
    lottie_dna = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
    if lottie_dna:
        st_lottie(lottie_dna, height=200, key="dna")

    # Initialize session state
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {'D00': [], 'D01': [], 'D02': [], 'D03': []}
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None

    # Instructions in an expander
    with st.expander("📋 How to Use This Platform", expanded=False):
        st.markdown("""
        1. **Upload Images**: Upload TIFF images for each time point (Day 0 to Day 3)
        2. **Verify**: Ensure each day has at least 1 image (typically 3-10 z-stack images)
        3. **Analyze**: Click 'Run DILITracer Analysis' to predict hepatotoxicity
        4. **Review**: Examine the AI predictions and risk assessment
        """)

    # Create tabs for better organization
    tab1, tab2 = st.tabs(["📤 Upload Images", "📊 Analysis Results"])

    with tab1:
        st.markdown("### Upload Organoid Images")
        
        # Progress indicator
        progress_cols = st.columns(4)
        upload_status = {}
        
        # Create columns for each day with better styling
        cols = st.columns(4)
        
        for idx, (day, col) in enumerate(zip(['D00', 'D01', 'D02', 'D03'], cols)):
            with col:
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background: #F3F4F6; border-radius: 10px;'>
                    <h4>Day {idx}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                files = st.file_uploader(
                    f"Select images",
                    type=['tiff', 'tif'],
                    accept_multiple_files=True,
                    key=f"upload_{day.lower()}"
                )
                
                if files:
                    st.session_state.uploaded_files[day] = files
                    upload_status[day] = len(files)
                    st.success(f"✅ {len(files)} images")
                    
                    # Show preview of first image
                    if st.checkbox(f"Preview", key=f"preview_{day}"):
                        image = Image.open(files[0])
                        st.image(image, caption=files[0].name, use_column_width=True)
                else:
                    upload_status[day] = 0
                    st.info("No images yet")
        
        # Upload summary with progress bar
        st.markdown("---")
        total_days_uploaded = sum(1 for v in upload_status.values() if v > 0)
        progress = total_days_uploaded / 4
        
        st.markdown("### 📊 Upload Progress")
        progress_bar = st.progress(progress)
        st.metric("Total Images Uploaded", sum(upload_status.values()))
        
        all_days_have_images = all(len(files) > 0 for files in st.session_state.uploaded_files.values())
        
        if st.button("🚀 Run DILITracer Analysis", disabled=not all_days_have_images, type="primary"):
            if all_days_have_images:
                with st.spinner("🧬 Initializing DILITracer model..."):
                    if st.session_state.analyzer is None:
                        st.session_state.analyzer = DILITracerAnalyzer()
                    time.sleep(1)
                
                with st.spinner("🔬 Analyzing organoid morphology changes..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    
                    # Run actual analysis
                    results = st.session_state.analyzer.run_analysis(st.session_state.uploaded_files)
                    st.session_state.analysis_results = results
                
                st.success("✅ Analysis Complete!")
                st.balloons()
                st.info("Go to 'Analysis Results' tab to view the predictions")

    with tab2:
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Create visually appealing results display
            st.markdown("## 🎯 DILITracer Prediction Results")
            
            # Main prediction display
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Color code based on risk
                risk_colors = {
                    "No-DILI": "#10B981",  # Green
                    "Less-DILI": "#F59E0B",  # Orange
                    "Most-DILI": "#EF4444"  # Red
                }
                
                color = risk_colors.get(results['prediction'], "#6B7280")
                
                st.markdown(f"""
                <div style='padding: 2rem; border-radius: 15px; background: {color}; color: white; text-align: center;'>
                    <h2 style='margin: 0;'>Predicted Classification</h2>
                    <h1 style='font-size: 3rem; margin: 0.5rem 0;'>{results['prediction']}</h1>
                    <h3 style='margin: 0;'>{results['risk_level']}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Probability visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = results['confidence'] * 100 if results['confidence'] > 0 else abs(results['confidence']) * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Score (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgray"},
                            {'range': [33, 66], 'color': "gray"},
                            {'range': [66, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Probability bar chart
                prob_data = {
                    'Category': ['No-DILI', 'Less-DILI', 'Most-DILI'],
                    'Probability': [
                        results['probabilities']['No-DILI'],
                        results['probabilities']['Less-DILI'],
                        results['probabilities']['Most-DILI']
                    ]
                }
                
                fig = px.bar(
                    prob_data, 
                    x='Category', 
                    y='Probability',
                    title="Classification Probabilities",
                    color='Category',
                    color_discrete_map={
                        'No-DILI': '#10B981',
                        'Less-DILI': '#F59E0B',
                        'Most-DILI': '#EF4444'
                    }
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation section
            st.markdown("### 📝 Clinical Interpretation")
            
            interpretations = {
                "No-DILI": {
                    "icon": "✅",
                    "text": "The compound shows **no significant hepatotoxic effects**. The organoids maintained healthy morphology throughout the treatment period.",
                    "recommendation": "Safe to proceed with further testing"
                },
                "Less-DILI": {
                    "icon": "⚠️",
                    "text": "The compound shows **moderate hepatotoxic effects**. Some morphological changes observed but organoids remain viable.",
                    "recommendation": "Proceed with caution, consider dose adjustment"
                },
                "Most-DILI": {
                    "icon": "🚫",
                    "text": "The compound shows **severe hepatotoxic effects**. Significant organoid damage and cell death observed.",
                    "recommendation": "High risk - reconsider compound or significantly modify"
                }
            }
            
            interp = interpretations[results['prediction']]
            st.info(f"{interp['icon']} {interp['text']}")
            st.warning(f"**Recommendation**: {interp['recommendation']}")
            
            # Download report button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("📥 Generate PDF Report", type="primary"):
                    with st.spinner("Generating report..."):
                        from report_generator import PDFReportGenerator
                        
                        generator = PDFReportGenerator()
                        pdf_path = generator.generate_report(results, st.session_state.uploaded_files)
                        
                        # Read the PDF file
                        with open(pdf_path, 'rb') as pdf_file:
                            pdf_bytes = pdf_file.read()
                        
                        # Create download button
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"DILITracer_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                        
                        st.success("✅ Report generated successfully!")
        
        else:
            # Empty state
            st.markdown("""
            <div style='text-align: center; padding: 4rem;'>
                <h2 style='color: #9CA3AF;'>No Analysis Results Yet</h2>
                <p style='color: #9CA3AF;'>Please upload images and run analysis first</p>
            </div>
            """, unsafe_allow_html=True)

with workflow_tab2:
    st.markdown("## 🔬 Organoid Morphology Analysis")
    st.info("Compare organoid sizes between Control and Experimental conditions using AI-powered segmentation")
    
    # Import analyzer
    from morphology_analyzer import MorphologyAnalyzer
    
    # Initialize session state
    if 'morphology_analyzer' not in st.session_state:
        st.session_state.morphology_analyzer = None
    if 'morphology_results' not in st.session_state:
        st.session_state.morphology_results = None
    
    # Create tabs
    upload_tab, analysis_tab, results_tab = st.tabs(["📤 Upload Images", "🔬 Analysis", "📊 Results"])
    
    with upload_tab:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔵 Control Group")
            control_files = st.file_uploader(
                "Upload Control Images",
                type=['tiff', 'tif', 'png', 'jpg'],
                accept_multiple_files=True,
                key="control_upload"
            )
            if control_files:
                st.success(f"✅ {len(control_files)} control images uploaded")
                if st.checkbox("Preview Control", key="preview_control"):
                    img = Image.open(control_files[0])
                    st.image(img, caption=control_files[0].name, use_column_width=True)
        
        with col2:
            st.markdown("### 🔴 Experimental Group")
            experimental_files = st.file_uploader(
                "Upload Experimental Images",
                type=['tiff', 'tif', 'png', 'jpg'],
                accept_multiple_files=True,
                key="experimental_upload"
            )
            if experimental_files:
                st.success(f"✅ {len(experimental_files)} experimental images uploaded")
                if st.checkbox("Preview Experimental", key="preview_exp"):
                    img = Image.open(experimental_files[0])
                    st.image(img, caption=experimental_files[0].name, use_column_width=True)
        
        # Settings
        st.markdown("---")
        st.markdown("### ⚙️ Analysis Settings")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pixel_to_um = st.number_input(
                "Pixel to μm conversion",
                min_value=0.1,
                max_value=10.0,
                value=0.65,
                step=0.05,
                help="Conversion factor from pixels to micrometers"
            )
        
        with col2:
            min_diameter = st.slider(
                "Minimum diameter filter (μm)",
                min_value=0,
                max_value=500,
                value=50,
                step=10,
                help="Only count organoids larger than this diameter"
            )
        
        with col3:
            st.metric("Total Images", len(control_files or []) + len(experimental_files or []))
    
    with analysis_tab:
        if control_files and experimental_files:
            if st.button("🚀 Run Morphology Analysis", type="primary"):
                with st.spinner("Initializing Cellpose model..."):
                    if st.session_state.morphology_analyzer is None:
                        st.session_state.morphology_analyzer = MorphologyAnalyzer()
                    
                    analyzer = st.session_state.morphology_analyzer
                    analyzer.pixel_to_um = pixel_to_um
                
                # Process control images
                control_measurements = []
                control_overlays = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(control_files):
                    status_text.text(f"Processing control image {idx+1}/{len(control_files)}")
                    
                    with st.spinner(f"Segmenting {file.name}..."):
                        masks, img = analyzer.segment_organoids(file)
                        df = analyzer.measure_organoids(masks, min_diameter)
                        control_measurements.append(df)
                        
                        overlay = analyzer.create_overlay(img, masks, df)
                        control_overlays.append(overlay)
                    
                    progress_bar.progress((idx + 1) / (len(control_files) + len(experimental_files)))
                
                # Process experimental images
                exp_measurements = []
                exp_overlays = []
                
                for idx, file in enumerate(experimental_files):
                    status_text.text(f"Processing experimental image {idx+1}/{len(experimental_files)}")
                    
                    with st.spinner(f"Segmenting {file.name}..."):
                        masks, img = analyzer.segment_organoids(file)
                        df = analyzer.measure_organoids(masks, min_diameter)
                        exp_measurements.append(df)
                        
                        overlay = analyzer.create_overlay(img, masks, df)
                        exp_overlays.append(overlay)
                    
                    progress_bar.progress((len(control_files) + idx + 1) / (len(control_files) + len(experimental_files)))
                
                # Combine measurements
                control_df = pd.concat(control_measurements, ignore_index=True) if control_measurements else pd.DataFrame()
                exp_df = pd.concat(exp_measurements, ignore_index=True) if exp_measurements else pd.DataFrame()
                
                # Perform comparison
                status_text.text("Performing statistical analysis...")
                results = analyzer.compare_conditions(control_df, exp_df)
                
                # Store results
                st.session_state.morphology_results = {
                    'control_df': control_df,
                    'exp_df': exp_df,
                    'statistics': results,
                    'control_overlays': control_overlays,
                    'exp_overlays': exp_overlays,
                    'min_diameter': min_diameter
                }
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis complete!")
                st.success("Analysis completed successfully! Check the Results tab.")
                st.balloons()
        else:
            st.warning("Please upload both control and experimental images first")
    
    with results_tab:
        if st.session_state.morphology_results:
            results = st.session_state.morphology_results
            stats = results['statistics']
            
            # Summary metrics
            st.markdown("### 📊 Summary Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🔵 Control Group")
                st.metric("Organoid Count", stats['control_stats']['count'])
                st.metric("Mean Diameter (μm)", f"{stats['control_stats']['mean_diameter_um']:.1f}")
                st.metric("Std Dev (μm)", f"{stats['control_stats']['std_diameter_um']:.1f}")
            
            with col2:
                st.markdown("#### 🔴 Experimental Group")
                st.metric("Organoid Count", stats['experimental_stats']['count'])
                st.metric("Mean Diameter (μm)", f"{stats['experimental_stats']['mean_diameter_um']:.1f}")
                st.metric("Std Dev (μm)", f"{stats['experimental_stats']['std_diameter_um']:.1f}")
            
            with col3:
                st.markdown("#### 📈 Statistical Test")
                if stats['ttest']['p_value'] is not None:
                    st.metric("T-statistic", f"{stats['ttest']['t_statistic']:.3f}")
                    st.metric("P-value", f"{stats['ttest']['p_value']:.4f}")
                    if stats['ttest']['significant']:
                        st.success("✅ Statistically Significant (p < 0.05)")
                    else:
                        st.info("❌ Not Significant (p ≥ 0.05)")
                else:
                    st.warning("Insufficient data for t-test")
            
            # Visualizations
            st.markdown("---")
            st.markdown("### 📊 Data Visualizations")
            
            # Size distribution comparison
            fig_dist = go.Figure()
            
            if len(results['control_df']) > 0:
                fig_dist.add_trace(go.Histogram(
                    x=results['control_df']['diameter_um'],
                    name='Control',
                    opacity=0.7,
                    marker_color='blue',
                    nbinsx=20
                ))
            
            if len(results['exp_df']) > 0:
                fig_dist.add_trace(go.Histogram(
                    x=results['exp_df']['diameter_um'],
                    name='Experimental',
                    opacity=0.7,
                    marker_color='red',
                    nbinsx=20
                ))
            
            fig_dist.update_layout(
                title="Organoid Size Distribution",
                xaxis_title="Diameter (μm)",
                yaxis_title="Count",
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Box plot comparison
            box_data = []
            if len(results['control_df']) > 0:
                for d in results['control_df']['diameter_um']:
                    box_data.append({'Group': 'Control', 'Diameter (μm)': d})
            if len(results['exp_df']) > 0:
                for d in results['exp_df']['diameter_um']:
                    box_data.append({'Group': 'Experimental', 'Diameter (μm)': d})
            
            if box_data:
                fig_box = px.box(
                    pd.DataFrame(box_data),
                    x='Group',
                    y='Diameter (μm)',
                    color='Group',
                    color_discrete_map={'Control': 'blue', 'Experimental': 'red'},
                    title="Organoid Size Comparison"
                )
                fig_box.update_layout(height=400)
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Segmentation results
            st.markdown("---")
            st.markdown("### 🔬 Segmentation Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if results['control_overlays']:
                    st.markdown("#### Control Segmentation")
                    selected_control = st.selectbox(
                        "Select control image",
                        range(len(results['control_overlays'])),
                        format_func=lambda x: f"Image {x+1}"
                    )
                    st.image(results['control_overlays'][selected_control], 
                            caption=f"Control Image {selected_control+1} - Segmented",
                            use_column_width=True)
            
            with col2:
                if results['exp_overlays']:
                    st.markdown("#### Experimental Segmentation")
                    selected_exp = st.selectbox(
                        "Select experimental image",
                        range(len(results['exp_overlays'])),
                        format_func=lambda x: f"Image {x+1}"
                    )
                    st.image(results['exp_overlays'][selected_exp],
                            caption=f"Experimental Image {selected_exp+1} - Segmented",
                            use_column_width=True)
            
            # Download results
            st.markdown("---")
            if st.button("📥 Download Results as CSV"):
                # Combine data with labels
                control_export = results['control_df'].copy()
                control_export['Group'] = 'Control'
                exp_export = results['exp_df'].copy()
                exp_export['Group'] = 'Experimental'
                
                combined_df = pd.concat([control_export, exp_export], ignore_index=True)
                
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"morphology_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No results yet. Please run analysis first.")