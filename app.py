import gradio as gr
import pandas as pd
from gait_analysis_processor import (
    setup_database, add_user, get_user, hash_password,
    check_password, load_user_history, analyze_gait,
    initialize_models
)

# --- APP LOGIC FUNCTIONS ---
def signup(username, password):
    if not username or not password:
        raise gr.Error("Username and password cannot be empty.")
    if len(password) < 6:
        raise gr.Error("Password must be at least 6 characters long.")
    hashed_pw = hash_password(password)
    if add_user(username, hashed_pw):
        return gr.Info("✅ Signup successful! You can now log in.")
    else:
        raise gr.Error("Username already exists. Please choose another.")

def login(username, password, state):
    if not username or not password:
        raise gr.Error("Username and password cannot be empty.")
    user_record = get_user(username)
    if user_record and check_password(password, user_record[0]):
        state["logged_in"] = True
        state["username"] = username
        user_history = load_user_history(username)
        return (
            state, 
            gr.update(visible=False), 
            gr.update(visible=True), 
            user_history,
            f"Welcome back, {username}! 👋"
        )
    else:
        raise gr.Error("❌ Invalid username or password.")

def logout(state):
    username = state.get("username", "User")
    state["logged_in"] = False
    state["username"] = None
    return (
        state, 
        gr.update(visible=True), 
        gr.update(visible=False), 
        None, "", {}, None, None, None, None, None,
        f"Goodbye, {username}! Come back soon. 👋"
    )

def run_analysis_for_user(video_file, prosthetic_side, state, progress=gr.Progress(track_tqdm=True)):
    if not state["logged_in"]:
        raise gr.Error("🔒 You must be logged in to run an analysis.")
    if video_file is None:
        raise gr.Error("📹 Please upload a video file first.")
    
    # Handle video file path
    if hasattr(video_file, 'name'):
        video_path = video_file.name
    else:
        video_path = video_file
    
    def progress_callback(fraction):
        progress(fraction, desc=f"🔍 Analyzing gait... {int(fraction*100)}%")
    
    # Run analysis
    video_out_path, summary, df, knee_fig, pelvic_fig, metrics_fig = analyze_gait(
        state["username"], video_path, prosthetic_side, progress_callback
    )
    
    if video_out_path is None:
        raise gr.Error("❌ Analysis failed. Please check your video file.")
    
    # Format summary text with enhanced metrics
    anomaly_indicator = "🚨 ANOMALY DETECTED" if summary['anomaly_detected'] == 1 else "✅ No Anomalies"
    
    summary_text = f"""
🤖 **Deep Learning Analysis Results**
**AI Pattern Recognition:**
• Gait Pattern: {summary['pattern_type']}
• Model Confidence: {summary['dl_confidence']}%
• Anomaly Status: {anomaly_indicator}
**Step Analysis:**
• Total Steps Detected: {summary['total_steps']}
• Cadence: {summary['cadence']:.1f} steps/min {"✅" if 90 <= summary['cadence'] <= 120 else "⚠️"}
• Balance Score: {summary['balance_score']:.1f}/100 {"✅" if summary['balance_score'] > 70 else "⚠️"}
**Joint Angles:**
• Left Knee (Avg): {summary['avg_left_knee_angle']:.1f}°
• Right Knee (Avg): {summary['avg_right_knee_angle']:.1f}°
• Knee Asymmetry: {summary['asymmetry']:.1f}° {"✅" if summary['asymmetry'] < 10 else "⚠️"}
**Pelvic Stability:**
• Average Pelvic Tilt: {summary['avg_pelvic_tilt']:.1f}°
• Pelvic Tilt Range: {summary['pelvic_tilt_range']:.1f}° {"✅" if summary['pelvic_tilt_range'] < 10 else "⚠️"}
**Gait Quality Metrics:**
• Avg Stride Length: {summary['avg_stride_length_px']:.1f} pixels
• Stride Variability: {summary['stride_variability']:.1f} {"✅" if summary['stride_variability'] < 15 else "⚠️"}
• Step Width: {summary['avg_step_width']:.1f} pixels
• Min Toe Clearance: {summary['toe_clearance_min']:.1f} pixels {"🚨" if summary['toe_clearance_min'] < 2 else "✅"}
• Gait Smoothness: {summary['gait_smoothness']:.2f} (lower is better)
**Risk Assessment:**
• Fall Risk Score: {summary['final_risk_score']}/100
• Risk Category: {summary['fall_risk']}
**Detected Issues:**
{summary['detected_issues']}
"""
    
    # Update user history
    user_history = load_user_history(state["username"])
    
    # Create risk label - simplified format for Gradio compatibility
    risk_emoji_map = {
        "Very Low": "✅", 
        "Low": "✔️", 
        "Moderate": "⚠️", 
        "High": "🚨"
    }
    
    # Gradio Label expects a dictionary with label:confidence pairs
    fall_risk_label = {
        f"{risk_emoji_map.get(summary['fall_risk'], '❓')} {summary['fall_risk']} Risk": summary['final_risk_score'] / 100,
        f"Pattern: {summary['pattern_type']}": summary['dl_confidence'] / 100
    }
    
    success_msg = f"✅ Analysis complete! Pattern: {summary['pattern_type']} | {summary['total_steps']} steps detected"
    
    return (
        video_out_path, 
        summary_text, 
        fall_risk_label, 
        df, 
        knee_fig, 
        pelvic_fig,
        metrics_fig,
        user_history,
        success_msg
    )

# --- GRADIO UI ---
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
    max-width: 1400px;
}
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem;
    border-radius: 15px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
.info-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-left: 5px solid #0284c7;
    padding: 1.2rem;
    border-radius: 8px;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.warning-box {
    background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
    border-left: 5px solid #f97316;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.info-box, .warning-box {
    font-size: 0.95rem;
    color: #374151;
}
.feature-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="NGPME - AI Gait Analysis") as app:
    
    gr.HTML("""
        <div class="main-header">
            <h1>🤖 Next-Gen Prosthetic Mobility Enhancer (NGPME)</h1>
            <h3>Advanced AI-Powered Gait Analysis</h3>
            <p>Deep Learning | Real-Time Feedback | Anomaly Detection</p>
        </div>
    """)
    
    auth_state = gr.State({"logged_in": False, "username": None})
    status_message = gr.Textbox(label="Status", visible=False, interactive=False)
    
    # --- LOGIN/SIGNUP UI ---
    with gr.Row(visible=True) as login_ui:
        with gr.Column(scale=1):
            gr.Markdown("## 🔐 Login")
            gr.HTML("""
                <div class="info-box">
                    <strong>Existing Users</strong><br>
                    Access your personalized dashboard and track progress over time
                </div>
            """)
            login_user = gr.Textbox(label="Username", placeholder="Enter your username")
            login_pass = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
            login_btn = gr.Button("🚀 Login", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("## 📝 Sign Up")
            gr.HTML("""
                <div class="info-box">
                    <strong>New Users</strong><br>
                    Create an account to start analyzing your gait patterns
                </div>
            """)
            signup_user = gr.Textbox(label="Choose Username", placeholder="Choose a unique username")
            signup_pass = gr.Textbox(label="Choose Password", type="password", placeholder="Min. 6 characters")
            signup_btn = gr.Button("✨ Sign Up", variant="secondary", size="lg")
    
    # --- MAIN APPLICATION UI ---
    with gr.Row(visible=False) as main_app_ui:
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Configuration")
            
            gr.HTML("""
                <div class="info-box">
                    <strong>📹 Video Guidelines:</strong><br>
                    ✓ Side view of walking (perpendicular to camera)<br>
                    ✓ Good lighting, clear subject visibility<br>
                    ✓ At least 5-10 steps captured<br>
                    ✓ Full body visible in frame<br>
                    ✓ Plain background recommended<br>
                    ✓ MP4, AVI, or MOV format
                </div>
            """)
            
            video_input = gr.Video(label="Upload Walking Video", height=300)
            
            prosthetic_side_input = gr.Dropdown(
                ["Left", "Right"], 
                label="Prosthetic Limb Side", 
                value="Right",
                info="Select which leg has the prosthetic device"
            )
            
            gr.HTML("""
                <div class="feature-card">
                    <strong>🤖 AI Features:</strong><br>
                    • LSTM Deep Learning Model<br>
                    • Real-time anomaly detection<br>
                    • Pattern recognition (Normal/Asymmetric/Shuffling/High-Risk)<br>
                    • Live issue feedback during analysis
                </div>
            """)
            
            analyze_button = gr.Button("🔍 Analyze Gait with AI", variant="primary", size="lg")
            
            gr.Markdown("---")
            logout_btn = gr.Button("🚪 Logout", variant="secondary")
            
        with gr.Column(scale=2):
            gr.Markdown("## 📊 AI Analysis Results")
            
            with gr.Tabs():
                with gr.TabItem("📈 Summary & Video"):
                    analysis_status = gr.Textbox(label="Analysis Status", interactive=False)
                    
                    fall_risk_output = gr.Label(
                        label="🎯 AI Risk Assessment", 
                        num_top_classes=3,
                        show_label=True
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            video_output = gr.Video(label="Annotated Gait Video with Real-Time Feedback")
                            
                            gr.HTML("""
                                <div class="warning-box">
                                    <strong>📺 Video Features:</strong><br>
                                    • Enhanced HUD with live metrics<br>
                                    • Real-time issue alerts<br>
                                    • Color-coded angle indicators<br>
                                    • Stride length visualization<br>
                                    • Center of mass tracking trail<br>
                                    • Pulsing step contact indicators
                                </div>
                            """)
                            
                        with gr.Column(scale=1):
                            summary_output = gr.Textbox(
                                label="Detailed AI Analysis", 
                                lines=28, 
                                interactive=False,
                                show_copy_button=True
                            )
                    
                    with gr.Accordion("🔬 Raw Frame Data", open=False):
                        detail_df_output = gr.DataFrame(
                            label="Frame-by-Frame Analysis",
                            wrap=True,
                            interactive=False
                        )
                
                with gr.TabItem("📉 Performance Charts"):
                    gr.HTML("""
                        <div class="info-box">
                            <strong>📊 Interactive Visualizations</strong><br>
                            Hover over charts for detailed data points. Zoom and pan for closer inspection.
                        </div>
                    """)
                    
                    knee_plot_output = gr.Plot(label="🦵 Knee Angle Dynamics with Step Detection")
                    
                    gr.Markdown("""
                    **Interpretation:**
                    - **Green zone (150-180°)**: Good extension
                    - **Yellow zone (60-150°)**: Normal flexion during swing
                    - **Red zone (<60°)**: Abnormal deep flexion
                    - **Vertical lines**: Detected steps (L=Left, R=Right)
                    """)
                    
                    pelvic_plot_output = gr.Plot(label="⚖️ Pelvic Stability Analysis")
                    
                    gr.Markdown("""
                    **Interpretation:**
                    - **Near 90°**: Level pelvis (ideal)
                    - **>10° deviation**: Indicates hip weakness or compensation
                    - **High variability**: Balance/stability concerns
                    """)
                    
                    metrics_plot_output = gr.Plot(label="👣 Stride Pattern Consistency")
                    
                    gr.Markdown("""
                    **Interpretation:**
                    - **Low variability**: Consistent, predictable gait
                    - **High variability**: Irregular steps, potential fall risk
                    - **Trend analysis**: Track improvement over sessions
                    """)
                
                with gr.TabItem("📜 Session History"):
                    gr.Markdown("### 📊 Your Analysis History")
                    
                    gr.HTML("""
                        <div class="info-box">
                            <strong>Track Your Progress</strong><br>
                            Compare metrics across sessions to monitor improvements and identify trends.
                        </div>
                    """)
                    
                    history_output = gr.DataFrame(
                        label="Past Sessions",
                        wrap=True,
                        interactive=False
                    )
                    
                    gr.Markdown("""
                    **Key Metrics to Track:**
                    - 📉 Fall risk score trend
                    - 📊 Cadence improvements
                    - ⚖️ Asymmetry reduction
                    - 👣 Stride consistency
                    """)
                
                with gr.TabItem("ℹ️ About & Help"):
                    gr.Markdown("""
                    ## 🤖 About NGPME AI System
                    
                    ### Technology Stack
                    - **LSTM Deep Learning**: Temporal pattern recognition in gait sequences
                    - **Anomaly Detection**: Isolation Forest algorithm for unusual patterns
                    - **Real-Time Analysis**: Frame-by-frame issue detection
                    - **MediaPipe Pose**: Advanced pose estimation
                    
                    ### Analyzed Metrics (16 Total)
                    1. **Step Count & Cadence** - Walking rhythm
                    2. **Knee Angles** - Joint flexion patterns
                    3. **Asymmetry** - Left-right balance
                    4. **Pelvic Stability** - Core control
                    5. **Stride Length & Variability** - Consistency
                    6. **Step Width** - Balance confidence
                    7. **Toe Clearance** - Trip risk
                    8. **Gait Smoothness** - Movement quality
                    9. **Balance Score** - Center of mass stability
                    10. **Pattern Classification** - AI-identified gait type
                    
                    ### Risk Categories
                    - **Very Low (0-25)**: Excellent gait, minimal concerns
                    - **Low (25-50)**: Good gait with minor improvements possible
                    - **Moderate (50-75)**: Notable issues, intervention beneficial
                    - **High (75-100)**: Significant fall risk, urgent attention needed
                    
                    ### Real-Time Video Annotations
                    
                    #### Color Codes:
                    - 🟢 **Green**: Normal/Good (angles 150-180°, step contacts)
                    - 🟡 **Yellow**: Caution (angles 60-150°, prosthetic leg, pelvic line)
                    - 🔴 **Red**: Warning (abnormal angles <60°, critical issues)
                    - 🟣 **Magenta**: Center of mass marker
                    - 🟠 **Orange**: Stride length indicators
                    
                    #### HUD Elements:
                    - **Progress Bar**: Analysis completion
                    - **Step Counter**: Real-time step detection
                    - **Live Alerts**: Immediate issue feedback
                    - **Metrics Display**: Current cadence, stride, consistency
                    
                    #### Visual Indicators:
                    - **Pulsing Circles**: Active foot contact
                    - **Angle Arcs**: Joint flexion with degree labels
                    - **COM Trail**: Balance tracking over time
                    - **Stride Lines**: Distance between steps
                    
                    ### Detected Issues Explained
                    
                    - **🚨 CRITICAL**: Immediate attention required (low toe clearance <2px)
                    - **⚠️ WARNING**: Significant concern (high asymmetry >15°, excessive pelvic tilt)
                    - **⚡ CAUTION**: Monitor and improve (moderate issues)
                    - **✅ GOOD**: No issues detected
                    
                    ### Tips for Best Results
                    1. Record in good lighting
                    2. Keep camera stable (tripod recommended)
                    3. Walk naturally, don't perform for camera
                    4. Ensure full body visible throughout
                    5. Walk for at least 10 steps
                    6. Use consistent recording setup for comparing sessions
                    
                    ### Privacy & Data
                    - All data stored locally in SQLite database
                    - Videos processed on your machine
                    - No data sent to external servers
                    - Session history private to your account
                    
                    ### Limitations
                    - Not a medical diagnostic tool
                    - Consult healthcare professionals for treatment decisions
                    - Works best with side-view videos
                    - Accuracy depends on video quality
                    
                    ### Support
                    For technical issues or questions:
                    - Check video quality guidelines
                    - Ensure all dependencies installed
                    - Review troubleshooting documentation
                    """)
    
    # --- EVENT HANDLERS ---
    
    signup_btn.click(
        signup, 
        inputs=[signup_user, signup_pass], 
        outputs=None
    )
    
    login_btn.click(
        login, 
        inputs=[login_user, login_pass, auth_state], 
        outputs=[auth_state, login_ui, main_app_ui, history_output, analysis_status]
    )
    
    logout_btn.click(
        logout, 
        inputs=[auth_state], 
        outputs=[
            auth_state, login_ui, main_app_ui, 
            video_output, summary_output, fall_risk_output, 
            detail_df_output, knee_plot_output, pelvic_plot_output, 
            metrics_plot_output, history_output, analysis_status
        ]
    )
    
    analyze_button.click(
        run_analysis_for_user,
        inputs=[video_input, prosthetic_side_input, auth_state],
        outputs=[
            video_output, summary_output, fall_risk_output, 
            detail_df_output, knee_plot_output, pelvic_plot_output,
            metrics_plot_output, history_output, analysis_status
        ]
    )
    
    # --- FOOTER ---
    gr.HTML("""
        <div style="text-align: center; margin-top: 3rem; padding: 2rem; 
                    background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); 
                    border-radius: 10px;">
            <h3 style="color: #374151; margin-bottom: 0.5rem;">
                🤖 NGPME v3.0 - AI Edition
            </h3>
            <p style="color: #6b7280; font-size: 0.9rem;">
                <strong>Powered by:</strong> LSTM Deep Learning • Isolation Forest Anomaly Detection • MediaPipe Pose
            </p>
            <p style="color: #6b7280; font-size: 0.85rem; margin-top: 0.5rem;">
                Advanced AI System for Prosthetic Gait Analysis | Built for Better Mobility
            </p>
            <p style="color: #9ca3af; font-size: 0.75rem; margin-top: 1rem;">
                ⚠️ For research and mobility assessment only. Always consult healthcare professionals.
            </p>
        </div>
    """)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 NGPME - Advanced AI Gait Analysis System")
    print("=" * 60)
    
    setup_database()
    print("✅ Database initialized")
    
    print("🧠 Initializing AI models (this may take a moment)...")
    initialize_models()
    print("✅ Deep Learning models loaded successfully!")
    
    print("\n" + "=" * 60)
    print("🌐 Launching web application...")
    print("=" * 60)
    
    app.launch(
        share=True, 
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=True
    )
