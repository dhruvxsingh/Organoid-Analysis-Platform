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

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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
    # Remove balloon animation, keep it clean
    st.markdown("## 🧬 Multi-Drug DILITracer Analysis")
    st.info("Compare hepatotoxicity across multiple drug candidates simultaneously")

    # Initialize session state for multiple conditions
    if 'multi_uploaded_files' not in st.session_state:
        st.session_state.multi_uploaded_files = {
            'Control': {'D00': [], 'D01': [], 'D02': [], 'D03': []},
            'Drug A': {'D00': [], 'D01': [], 'D02': [], 'D03': []},
            'Drug B': {'D00': [], 'D01': [], 'D02': [], 'D03': []},
            'Drug C': {'D00': [], 'D01': [], 'D02': [], 'D03': []}
        }
    if 'multi_analysis_results' not in st.session_state:
        st.session_state.multi_analysis_results = {}
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None

    # Instructions
    with st.expander("📋 How to Use Multi-Drug Analysis", expanded=False):
        st.markdown("""
        1. **Upload Images**: For each condition (Control, Drug A/B/C), upload 4-day image series
        2. **Flexible Analysis**: Upload only the conditions you want to compare
        3. **Run Analysis**: DILITracer will analyze each uploaded condition
        4. **Compare Results**: View side-by-side toxicity predictions
        5. **Generate Report**: Download comprehensive comparison report
        """)

    # Create tabs
    tab1, tab2 = st.tabs(["📤 Upload Images", "📊 Comparison Results"])

    with tab1:
        st.markdown("### 📁 Upload Images for Each Condition")
        
        # Quick load demo button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Load Demo Images (Quick Test)", type="secondary"):
                # This will load pre-selected images
                st.info("Demo feature: Would load pre-selected images with expected results:\n"
                       "• Control → No-DILI\n"
                       "• Drug A → Less-DILI\n" 
                       "• Drug B → No-DILI\n"
                       "• Drug C → Most-DILI")
                # TODO: Implement actual demo loading
        
        st.markdown("---")
        
        # Create upload sections for each condition
        conditions = ['Control', 'Drug A', 'Drug B', 'Drug C']
        condition_colors = {
            'Control': '#10B981',  # Green
            'Drug A': '#3B82F6',   # Blue
            'Drug B': '#8B5CF6',   # Purple
            'Drug C': '#EF4444'    # Red
        }
        
        upload_status = {}
        
        for condition in conditions:
            with st.expander(f"**{condition}** - Upload 4-day Image Series", expanded=True):
                # Create 4 columns for each day
                cols = st.columns(4)
                condition_status = {}
                
                for idx, (day, col) in enumerate(zip(['D00', 'D01', 'D02', 'D03'], cols)):
                    with col:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 0.5rem; 
                                    background: {condition_colors[condition]}20; 
                                    border-radius: 8px; margin-bottom: 0.5rem;'>
                            <b>Day {idx}</b>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        files = st.file_uploader(
                            f"Upload",
                            type=['tiff', 'tif'],
                            accept_multiple_files=True,
                            key=f"upload_{condition}_{day}".replace(" ", "_"),
                            label_visibility="collapsed"
                        )
                        
                        if files:
                            st.session_state.multi_uploaded_files[condition][day] = files
                            condition_status[day] = len(files)
                            st.success(f"✅ {len(files)}")
                        else:
                            condition_status[day] = 0
                            st.caption("No images")
                
                # Check if this condition has all days uploaded
                all_days = all(condition_status.get(f'D0{i}', 0) > 0 for i in range(4))
                upload_status[condition] = {
                    'complete': all_days,
                    'total_images': sum(condition_status.values())
                }
                
                if all_days:
                    st.success(f"✅ {condition} ready for analysis ({upload_status[condition]['total_images']} images)")
                elif upload_status[condition]['total_images'] > 0:
                    st.warning(f"⚠️ {condition} incomplete - some days missing")
                else:
                    st.info(f"No images uploaded for {condition}")
        
        # Summary and run button
        st.markdown("---")
        st.markdown("### 📊 Upload Summary")
        
        # Show which conditions are ready
        ready_conditions = [c for c, status in upload_status.items() if status['complete']]
        partial_conditions = [c for c, status in upload_status.items() 
                            if status['total_images'] > 0 and not status['complete']]
        
        col1, col2 = st.columns(2)
        with col1:
            if ready_conditions:
                st.success(f"**Ready for analysis:** {', '.join(ready_conditions)}")
            else:
                st.warning("No conditions ready for analysis")
        
        with col2:
            if partial_conditions:
                st.warning(f"**Incomplete:** {', '.join(partial_conditions)}")
        
        # Run analysis button
        if ready_conditions:
            if st.button(f"🚀 Run DILITracer Analysis on {len(ready_conditions)} Condition(s)", 
                        type="primary", use_container_width=True):
                
                with st.spinner("🧬 Initializing DILITracer model..."):
                    if st.session_state.analyzer is None:
                        st.session_state.analyzer = DILITracerAnalyzer()
                
                # Clear previous results
                st.session_state.multi_analysis_results = {}
                
                # Analyze each ready condition
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, condition in enumerate(ready_conditions):
                    status_text.text(f"Analyzing {condition}...")
                    
                    # Get files for this condition
                    condition_files = st.session_state.multi_uploaded_files[condition]
                    
                    # Run analysis
                    results = st.session_state.analyzer.run_analysis(condition_files)
                    st.session_state.multi_analysis_results[condition] = results
                    
                    progress_bar.progress((idx + 1) / len(ready_conditions))
                
                status_text.text("✅ Analysis complete!")
                st.success(f"Successfully analyzed {len(ready_conditions)} conditions!")
                st.info("Go to 'Comparison Results' tab to view the analysis")
        else:
            st.warning("Please upload complete 4-day series for at least one condition")

    with tab2:
        if st.session_state.multi_analysis_results:
            results = st.session_state.multi_analysis_results
            
            st.markdown("## 🎯 DILITracer Comparison Results")
            st.markdown(f"**Analyzed Conditions:** {', '.join(results.keys())}")
            
            # Summary cards showing all conditions
            st.markdown("### 📊 Toxicity Summary")
            
            cols = st.columns(len(results))
            
            for col, (condition, result) in zip(cols, results.items()):
                with col:
                    # Color based on prediction
                    color = {
                        "No-DILI": "#10B981",
                        "Less-DILI": "#F59E0B", 
                        "Most-DILI": "#EF4444"
                    }.get(result['prediction'], "#6B7280")
                    
                    # Create result card
                    st.markdown(f"""
                    <div style='padding: 1.5rem; border-radius: 10px; 
                               background: {color}; color: white; text-align: center;'>
                        <h3 style='margin: 0;'>{condition}</h3>
                        <h2 style='margin: 0.5rem 0;'>{result['prediction']}</h2>
                        <p style='margin: 0;'>{result['risk_level']}</p>
                        <p style='margin: 0.5rem 0; font-size: 0.9rem;'>
                            Confidence: {result['confidence']*100:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Recommendation icon
                    if result['prediction'] == "Most-DILI":
                        st.error("❌ ELIMINATE - High toxicity")
                    elif result['prediction'] == "Less-DILI":
                        st.warning("⚠️ CAUTION - Moderate toxicity")
                    else:
                        st.success("✅ SAFE - Low toxicity")
            
            st.markdown("---")
            
            # Detailed comparison table
            st.markdown("### 📈 Detailed Probability Comparison")
            
            # Create comparison dataframe
            comparison_data = []
            for condition, result in results.items():
                comparison_data.append({
                    'Condition': condition,
                    'Prediction': result['prediction'],
                    'No-DILI (%)': f"{result['probabilities']['No-DILI']*100:.2f}",
                    'Less-DILI (%)': f"{result['probabilities']['Less-DILI']*100:.2f}",
                    'Most-DILI (%)': f"{result['probabilities']['Most-DILI']*100:.2f}",
                    'Risk Level': result['risk_level']
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Style the dataframe
            def style_prediction(val):
                if val == "Most-DILI":
                    return 'background-color: #FEE2E2'
                elif val == "Less-DILI":
                    return 'background-color: #FEF3C7'
                else:
                    return 'background-color: #D1FAE5'
            
            styled_df = df_comparison.style.applymap(style_prediction, subset=['Prediction'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Visualization - Grouped bar chart
            st.markdown("### 📊 Probability Distribution Comparison")
            
            # Prepare data for plotting
            categories = ['No-DILI', 'Less-DILI', 'Most-DILI']
            fig = go.Figure()
            
            colors_map = {
                'Control': '#10B981',
                'Drug A': '#3B82F6',
                'Drug B': '#8B5CF6',
                'Drug C': '#EF4444'
            }
            
            for condition, result in results.items():
                fig.add_trace(go.Bar(
                    name=condition,
                    x=categories,
                    y=[result['probabilities'][cat] for cat in categories],
                    marker_color=colors_map.get(condition, '#6B7280')
                ))
            
            fig.update_layout(
                barmode='group',
                title="Toxicity Probability Comparison",
                xaxis_title="DILI Category",
                yaxis_title="Probability",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Drug selection recommendation
            st.markdown("---")
            st.markdown("### 🎯 Drug Selection Recommendation")
            
            # Sort by safety (No-DILI probability)
            sorted_conditions = sorted(results.items(), 
                                     key=lambda x: x[1]['probabilities']['No-DILI'], 
                                     reverse=True)
            
            st.markdown("**Ranking by Safety (No-DILI Probability):**")
            for rank, (condition, result) in enumerate(sorted_conditions, 1):
                emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📊"
                st.markdown(f"{emoji} **{rank}. {condition}** - "
                          f"No-DILI: {result['probabilities']['No-DILI']*100:.1f}% "
                          f"({result['prediction']})")
            
            # Generate report button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("📥 Generate Comparison Report", type="primary"):
                    with st.spinner("Generating comprehensive report..."):
                        from report_generator_multi import MultiDrugReportGenerator
                        
                        generator = MultiDrugReportGenerator()
                        pdf_path = generator.generate_comparison_report(
                            results, 
                            st.session_state.multi_uploaded_files
                        )
                        
                        with open(pdf_path, 'rb') as pdf_file:
                            pdf_bytes = pdf_file.read()
                        
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"Multi_Drug_Comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                        
                        st.success("✅ Report generated successfully!")
        else:
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