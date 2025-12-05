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
    st.info("Count and measure organoids with AI-powered segmentation. Compare Control vs Experimental conditions.")
    
    # Import analyzer
    from morphology_v3 import OrganoidAnalyzer
    
    # Initialize session state
    if 'morphology_analyzer' not in st.session_state:
        st.session_state.morphology_analyzer = None
    if 'morphology_results' not in st.session_state:
        st.session_state.morphology_results = None
    
    # Create tabs
    upload_tab, results_tab, growth_tab = st.tabs([
        "📤 Upload & Analyze", 
        "📊 Results",
        "📈 Growth Tracking"
    ])
    
    with upload_tab:
        # Settings at the top
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
            min_size = st.slider(
                "Minimum object size (pixels)",
                min_value=100,
                max_value=2000,
                value=300,
                step=50,
                help="Filter out objects smaller than this (removes debris)"
            )
        
        with col3:
            min_diameter = st.slider(
                "Minimum diameter filter (μm)",
                min_value=0,
                max_value=200,
                value=0,
                step=10,
                help="Only report organoids larger than this diameter"
            )

        # Advanced settings (expandable)
        with st.expander("🔧 Advanced Segmentation Settings"):
            st.info("Adjust these if detection is missing organoids or detecting too many false positives")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                diameter_mode = st.radio(
                    "Diameter estimation",
                    ["Auto-detect", "Manual"],
                    help="Auto-detect works for most images. Use Manual if detection is poor."
                )
                
                if diameter_mode == "Manual":
                    manual_diameter = st.number_input(
                        "Expected diameter (pixels)",
                        min_value=20,
                        max_value=500,
                        value=100,
                        step=10,
                        help="Typical organoid diameter in your images"
                    )
                else:
                    manual_diameter = None
            
            with col2:
                flow_threshold = st.slider(
                    "Flow threshold",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    help="Lower = detect more objects (may include noise)"
                )
            
            with col3:
                cellprob_threshold = st.slider(
                    "Cell probability threshold",
                    min_value=-6,
                    max_value=2,
                    value=-4,
                    step=1,
                    help="Lower = detect fainter organoids"
                )
        
        st.markdown("---")
        
        # Upload section
        st.markdown("### 📁 Upload Images")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔵 Control Group")
            control_files = st.file_uploader(
                "Upload Control Images",
                type=['tiff', 'tif', 'png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="control_upload",
                help="Upload one or more organoid images for the control condition"
            )
            if control_files:
                st.success(f"✅ {len(control_files)} image(s) uploaded")
                # Preview first image
                with st.expander("Preview Control Image"):
                    img = Image.open(control_files[0])
                    st.image(img, caption=f"{control_files[0].name} (Original)", use_column_width=True)
        
        with col2:
            st.markdown("#### 🔴 Experimental Group")
            experimental_files = st.file_uploader(
                "Upload Experimental Images",
                type=['tiff', 'tif', 'png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="experimental_upload",
                help="Upload one or more organoid images for the experimental condition"
            )
            if experimental_files:
                st.success(f"✅ {len(experimental_files)} image(s) uploaded")
                # Preview first image
                with st.expander("Preview Experimental Image"):
                    img = Image.open(experimental_files[0])
                    st.image(img, caption=f"{experimental_files[0].name} (Original)", use_column_width=True)
        
        st.markdown("---")
        
        # Run Analysis
        if control_files or experimental_files:
            total_images = len(control_files or []) + len(experimental_files or [])
            st.info(f"📊 Ready to analyze {total_images} image(s)")
            
            if st.button("🚀 Run Morphology Analysis", type="primary", use_container_width=True):
                
                # Initialize analyzer
                with st.spinner("Initializing segmentation model..."):
                    if st.session_state.morphology_analyzer is None:
                        st.session_state.morphology_analyzer = OrganoidAnalyzer(method='classical')
                    
                    analyzer = st.session_state.morphology_analyzer
                    analyzer.pixel_to_um = pixel_to_um
                    analyzer.params['min_size_pixels'] = min_size
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_steps = len(control_files or []) + len(experimental_files or [])
                current_step = 0
                
                # Process Control images
                control_measurements = []
                control_overlays = []
                control_boundary_overlays = []
                
                if control_files:
                    for idx, file in enumerate(control_files):
                        status_text.text(f"🔵 Processing Control image {idx+1}/{len(control_files)}: {file.name}")
                        
                        try:
                            # Reset file pointer
                            file.seek(0)
                            
                            # Segment
                            masks, img = analyzer.segment_organoids(file, min_size=min_size)
                            
                            # Measure
                            df = analyzer.measure_organoids(masks, min_diameter_um=min_diameter)
                            df['source_image'] = file.name
                            control_measurements.append(df)
                            
                            # Create overlays
                            overlay = analyzer.create_overlay(img, masks, df)
                            control_overlays.append({
                                'name': file.name,
                                'image': overlay,
                                'count': len(df)
                            })
                            
                            boundary = analyzer.create_boundary_overlay(img, masks, df)
                            control_boundary_overlays.append({
                                'name': file.name,
                                'image': boundary,
                                'count': len(df)
                            })
                            
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                            continue
                        
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                
                # Process Experimental images
                exp_measurements = []
                exp_overlays = []
                exp_boundary_overlays = []
                
                if experimental_files:
                    for idx, file in enumerate(experimental_files):
                        status_text.text(f"🔴 Processing Experimental image {idx+1}/{len(experimental_files)}: {file.name}")
                        
                        try:
                            file.seek(0)
                            
                            masks, img = analyzer.segment_organoids(file, min_size=min_size)
                            df = analyzer.measure_organoids(masks, min_diameter_um=min_diameter)
                            df['source_image'] = file.name
                            exp_measurements.append(df)
                            
                            overlay = analyzer.create_overlay(img, masks, df)
                            exp_overlays.append({
                                'name': file.name,
                                'image': overlay,
                                'count': len(df)
                            })
                            
                            boundary = analyzer.create_boundary_overlay(img, masks, df)
                            exp_boundary_overlays.append({
                                'name': file.name,
                                'image': boundary,
                                'count': len(df)
                            })
                            
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                            continue
                        
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                
                # Combine measurements
                control_df = pd.concat(control_measurements, ignore_index=True) if control_measurements else pd.DataFrame()
                exp_df = pd.concat(exp_measurements, ignore_index=True) if exp_measurements else pd.DataFrame()
                
                # Statistical comparison
                status_text.text("📊 Performing statistical analysis...")
                stats_results = analyzer.compare_conditions(control_df, exp_df)
                
                # Store results
                st.session_state.morphology_results = {
                    'control_df': control_df,
                    'exp_df': exp_df,
                    'statistics': stats_results,
                    'control_overlays': control_overlays,
                    'exp_overlays': exp_overlays,
                    'control_boundary_overlays': control_boundary_overlays,
                    'exp_boundary_overlays': exp_boundary_overlays,
                    'settings': {
                        'pixel_to_um': pixel_to_um,
                        'min_size': min_size,
                        'min_diameter': min_diameter
                    }
                }
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis complete!")
                
                # Show summary
                st.success(f"""
                **Analysis Complete!**
                - Control: {len(control_df)} organoids detected
                - Experimental: {len(exp_df)} organoids detected
                
                Go to the **Results** tab to view detailed results and visualizations.
                """)
        else:
            st.warning("Please upload at least one image to analyze")
    
    with results_tab:
        if st.session_state.morphology_results:
            results = st.session_state.morphology_results
            stats = results['statistics']
            
            # ===== SUMMARY SECTION =====
            st.markdown("## 📊 Analysis Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🔵 Control")
                st.metric("Organoid Count", stats['control_stats']['count'])
                if stats['control_stats']['count'] > 0:
                    st.metric("Mean Diameter", f"{stats['control_stats']['mean_diameter_um']:.1f} μm")
                    st.metric("Std Dev", f"{stats['control_stats']['std_diameter_um']:.1f} μm")
            
            with col2:
                st.markdown("### 🔴 Experimental")
                st.metric("Organoid Count", stats['experimental_stats']['count'])
                if stats['experimental_stats']['count'] > 0:
                    st.metric("Mean Diameter", f"{stats['experimental_stats']['mean_diameter_um']:.1f} μm")
                    st.metric("Std Dev", f"{stats['experimental_stats']['std_diameter_um']:.1f} μm")
            
            with col3:
                st.markdown("### 📈 Statistics")
                if stats['ttest']['p_value'] is not None:
                    st.metric("T-statistic", f"{stats['ttest']['t_statistic']:.3f}")
                    st.metric("P-value", f"{stats['ttest']['p_value']:.4f}")
                    if stats['ttest']['significant']:
                        st.success("✅ Significant (p < 0.05)")
                    else:
                        st.info("Not Significant (p ≥ 0.05)")
                else:
                    st.warning("Insufficient data for t-test")
            
            st.markdown("---")
            
            # ===== SEGMENTATION VISUALIZATION =====
            st.markdown("## 🔬 Segmentation Results")
            st.info("View how organoids were detected and counted. Each colored boundary = one organoid.")
            
            # Visualization type selector
            viz_type = st.radio(
                "Visualization Style:",
                ["Boundaries Only (Clean)", "Full Overlay (with labels)"],
                horizontal=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔵 Control Segmentation")
                if results['control_overlays']:
                    overlays = results['control_boundary_overlays'] if "Boundaries" in viz_type else results['control_overlays']
                    
                    for item in overlays:
                        st.image(
                            item['image'],
                            caption=f"{item['name']} - {item['count']} organoids detected",
                            use_column_width=True
                        )
                else:
                    st.info("No control images analyzed")
            
            with col2:
                st.markdown("### 🔴 Experimental Segmentation")
                if results['exp_overlays']:
                    overlays = results['exp_boundary_overlays'] if "Boundaries" in viz_type else results['exp_overlays']
                    
                    for item in overlays:
                        st.image(
                            item['image'],
                            caption=f"{item['name']} - {item['count']} organoids detected",
                            use_column_width=True
                        )
                else:
                    st.info("No experimental images analyzed")
            
            st.markdown("---")
            
            # ===== DATA VISUALIZATIONS =====
            st.markdown("## 📊 Data Visualizations")
            
            # Only show if we have data
            has_control = len(results['control_df']) > 0
            has_exp = len(results['exp_df']) > 0
            
            if has_control or has_exp:
                # Histogram
                st.markdown("### Size Distribution")
                fig_hist = go.Figure()
                
                if has_control:
                    fig_hist.add_trace(go.Histogram(
                        x=results['control_df']['diameter_um'],
                        name='Control',
                        opacity=0.7,
                        marker_color='#3B82F6',
                        nbinsx=15
                    ))
                
                if has_exp:
                    fig_hist.add_trace(go.Histogram(
                        x=results['exp_df']['diameter_um'],
                        name='Experimental',
                        opacity=0.7,
                        marker_color='#EF4444',
                        nbinsx=15
                    ))
                
                fig_hist.update_layout(
                    title="Organoid Diameter Distribution",
                    xaxis_title="Diameter (μm)",
                    yaxis_title="Count",
                    barmode='overlay',
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Box plot
                if has_control and has_exp:
                    st.markdown("### Diameter Comparison")
                    
                    box_data = []
                    for d in results['control_df']['diameter_um']:
                        box_data.append({'Group': 'Control', 'Diameter (μm)': d})
                    for d in results['exp_df']['diameter_um']:
                        box_data.append({'Group': 'Experimental', 'Diameter (μm)': d})
                    
                    fig_box = px.box(
                        pd.DataFrame(box_data),
                        x='Group',
                        y='Diameter (μm)',
                        color='Group',
                        color_discrete_map={'Control': '#3B82F6', 'Experimental': '#EF4444'},
                        title="Organoid Size Comparison (Box Plot)"
                        points="all"
                    )
                    fig_box.update_traces(
                        jitter=0.3,
                        pointpos=0
                    ) 
                    fig_box.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_box, use_container_width=True)
            
            st.markdown("---")
            
            # ===== DETAILED DATA TABLE =====
            st.markdown("## 📋 Detailed Measurements")
            
            tab1, tab2 = st.tabs(["Control Data", "Experimental Data"])
            
            with tab1:
                if has_control:
                    display_cols = ['organoid_id', 'diameter_um', 'area_um2', 'circularity', 'source_image']
                    available_cols = [c for c in display_cols if c in results['control_df'].columns]
                    st.dataframe(results['control_df'][available_cols], use_container_width=True)
                else:
                    st.info("No control data")
            
            with tab2:
                if has_exp:
                    display_cols = ['organoid_id', 'diameter_um', 'area_um2', 'circularity', 'source_image']
                    available_cols = [c for c in display_cols if c in results['exp_df'].columns]
                    st.dataframe(results['exp_df'][available_cols], use_container_width=True)
                else:
                    st.info("No experimental data")
            
            # ===== DOWNLOAD =====
            st.markdown("---")
            st.markdown("## 📥 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Prepare combined data
                export_data = []
                if has_control:
                    ctrl = results['control_df'].copy()
                    ctrl['group'] = 'Control'
                    export_data.append(ctrl)
                if has_exp:
                    exp = results['exp_df'].copy()
                    exp['group'] = 'Experimental'
                    export_data.append(exp)
                
                if export_data:
                    combined = pd.concat(export_data, ignore_index=True)
                    # Remove centroid and bbox (not CSV friendly)
                    export_cols = [c for c in combined.columns if c not in ['centroid', 'bbox']]
                    csv = combined[export_cols].to_csv(index=False)
                    
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv,
                        file_name=f"morphology_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                # Summary report
                summary_text = f"""
ORGANOID MORPHOLOGY ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SETTINGS:
- Pixel to μm: {results['settings']['pixel_to_um']}
- Min object size: {results['settings']['min_size']} px
- Min diameter filter: {results['settings']['min_diameter']} μm

CONTROL GROUP:
- Count: {stats['control_stats']['count']}
- Mean Diameter: {stats['control_stats']['mean_diameter_um']:.2f} μm
- Std Dev: {stats['control_stats']['std_diameter_um']:.2f} μm

EXPERIMENTAL GROUP:
- Count: {stats['experimental_stats']['count']}
- Mean Diameter: {stats['experimental_stats']['mean_diameter_um']:.2f} μm
- Std Dev: {stats['experimental_stats']['std_diameter_um']:.2f} μm

STATISTICAL TEST (Welch's t-test):
- T-statistic: {stats['ttest']['t_statistic']}
- P-value: {stats['ttest']['p_value']}
- Significant: {'Yes' if stats['ttest']['significant'] else 'No'}
"""
                st.download_button(
                    label="📝 Download Summary",
                    data=summary_text,
                    file_name=f"morphology_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        else:
            st.markdown("""
            <div style='text-align: center; padding: 4rem;'>
                <h2 style='color: #9CA3AF;'>No Results Yet</h2>
                <p style='color: #9CA3AF;'>Upload images and run analysis to see results</p>
            </div>
            """, unsafe_allow_html=True)

    with growth_tab:
        st.markdown("## 📈 Organoid Growth Tracking")
        st.info("Track how organoid diameter changes over multiple days")

        # Use the same analyzer as the main analysis
        if 'growth_analyzer' not in st.session_state:
            st.session_state.growth_analyzer = None
        if 'growth_data' not in st.session_state:
            st.session_state.growth_data = {}

        # Settings (same as main tab)
        st.markdown("### ⚙️ Settings")
        col1, col2 = st.columns(2)
        with col1:
            growth_pixel_to_um = st.number_input(
                "Pixel to μm",
                min_value=0.1,
                max_value=10.0,
                value=0.65,
                step=0.05,
                key="growth_pixel"
            )
        with col2:
            growth_min_size = st.slider(
                "Min object size (pixels)",
                min_value=100,
                max_value=2000,
                value=300,
                step=50,
                key="growth_min_size"
            )

        st.markdown("---")
        st.markdown("### 📁 Upload Images for Each Day")

        # Simple 4-day upload
        col1, col2, col3, col4 = st.columns(4)

        day_files = {}

        with col1:
            st.markdown("**Day 1**")
            day1_num = st.number_input("Day number", value=0, key="day1_num", min_value=0)
            day1_file = st.file_uploader("Upload", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'], key="day1_file", label_visibility="collapsed")
            if day1_file:
                day_files[day1_num] = {'file': day1_file, 'name': f"Day {day1_num}"}
                st.success("✅")

        with col2:
            st.markdown("**Day 2**")
            day2_num = st.number_input("Day number", value=7, key="day2_num", min_value=0)
            day2_file = st.file_uploader("Upload", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'], key="day2_file", label_visibility="collapsed")
            if day2_file:
                day_files[day2_num] = {'file': day2_file, 'name': f"Day {day2_num}"}
                st.success("✅")

        with col3:
            st.markdown("**Day 3 (Optional)**")
            day3_num = st.number_input("Day number", value=14, key="day3_num", min_value=0)
            day3_file = st.file_uploader("Upload", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'], key="day3_file", label_visibility="collapsed")
            if day3_file:
                day_files[day3_num] = {'file': day3_file, 'name': f"Day {day3_num}"}
                st.success("✅")

        with col4:
            st.markdown("**Day 4 (Optional)**")
            day4_num = st.number_input("Day number", value=21, key="day4_num", min_value=0)
            day4_file = st.file_uploader("Upload", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'], key="day4_file", label_visibility="collapsed")
            if day4_file:
                day_files[day4_num] = {'file': day4_file, 'name': f"Day {day4_num}"}
                st.success("✅")

        st.markdown("---")

        # Analyze button
        if len(day_files) >= 2:
            st.success(f"✅ {len(day_files)} timepoints ready")
            
            if st.button("🚀 Analyze Growth", type="primary", use_container_width=True, key="growth_analyze_btn"):
                
                # Initialize analyzer (SAME as main tab)
                with st.spinner("Analyzing..."):
                    analyzer = OrganoidAnalyzer(method='classical')
                    analyzer.pixel_to_um = growth_pixel_to_um
                    analyzer.params['min_size_pixels'] = growth_min_size
                    
                    results = {}
                    
                    # Sort by day number
                    sorted_days = sorted(day_files.keys())
                    
                    progress = st.progress(0)
                    
                    for idx, day_num in enumerate(sorted_days):
                        data = day_files[day_num]
                        file = data['file']
                        
                        st.text(f"Processing Day {day_num}...")
                        
                        # Reset file pointer
                        file.seek(0)
                        
                        # Run SAME segmentation as main tab
                        masks, img = analyzer.segment_organoids(file, min_size=growth_min_size)
                        measurements = analyzer.measure_organoids(masks, min_diameter_um=0)
                        
                        # Create overlay (SAME as main tab)
                        overlay = analyzer.create_overlay(img, masks, measurements)
                        
                        # Store results
                        results[day_num] = {
                            'name': f"Day {day_num}",
                            'count': len(measurements),
                            'mean_diameter': measurements['diameter_um'].mean() if len(measurements) > 0 else 0,
                            'std_diameter': measurements['diameter_um'].std() if len(measurements) > 0 else 0,
                            'overlay': overlay,
                            'measurements': measurements
                        }
                        
                        progress.progress((idx + 1) / len(sorted_days))
                    
                    # Store in session state
                    st.session_state.growth_data = results
                
                st.success("✅ Analysis complete!")

        elif len(day_files) == 1:
            st.warning("Please upload at least 2 images to track growth")
        else:
            st.info("Upload images for at least 2 different days")

        # Display Results
        if st.session_state.growth_data:
            results = st.session_state.growth_data
            sorted_days = sorted(results.keys())
            
            st.markdown("---")
            st.markdown("## 📊 Results")
            
            # Calculate growth
            first_day = sorted_days[0]
            last_day = sorted_days[-1]
            initial_diameter = results[first_day]['mean_diameter']
            final_diameter = results[last_day]['mean_diameter']
            
            if initial_diameter > 0:
                growth_percent = ((final_diameter - initial_diameter) / initial_diameter) * 100
            else:
                growth_percent = 0
            
            # Summary metrics
            st.markdown("### 📈 Growth Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"Initial (Day {first_day})",
                    f"{initial_diameter:.1f} μm"
                )
            
            with col2:
                st.metric(
                    f"Final (Day {last_day})",
                    f"{final_diameter:.1f} μm"
                )
            
            with col3:
                st.metric(
                    "Total Growth",
                    f"{growth_percent:+.1f}%",
                    delta=f"{final_diameter - initial_diameter:+.1f} μm"
                )
            
            # Data table
            st.markdown("### 📋 Measurements by Day")
            
            table_data = []
            for day_num in sorted_days:
                data = results[day_num]
                table_data.append({
                    'Day': day_num,
                    'Organoid Count': data['count'],
                    'Mean Diameter (μm)': round(data['mean_diameter'], 1),
                    'Std Dev (μm)': round(data['std_diameter'], 1)
                })
            
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            
            # Show detected images
            st.markdown("### 🔬 Detected Images")
            
            cols = st.columns(len(sorted_days))
            
            for col, day_num in zip(cols, sorted_days):
                data = results[day_num]
                with col:
                    st.image(
                        data['overlay'],
                        caption=f"Day {day_num}\n{data['count']} organoids\nØ {data['mean_diameter']:.1f} μm",
                        use_column_width=True
                    )
            
            # Download
            st.markdown("---")
            csv_data = pd.DataFrame(table_data).to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                data=csv_data,
                file_name=f"growth_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )