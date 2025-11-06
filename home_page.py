import streamlit as st


def show_home_page():
    st.markdown(custom_css(), unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="container">
            <div class="header">🚀 AutoEDA</div>
            <div class="tagline">Automated Exploratory Data Analysis Made Simple</div>
        </div>
    """, unsafe_allow_html=True)

    # Key Features
    st.markdown("""
        <div class="section-title">✨ Key Features</div>
        <div class="features">
            <div class="feature">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Interactive Exploration</div>
                <p>Explore your datasets with dynamic, interactive visualizations.</p>
            </div>
            <div class="feature">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Stunning Charts</div>
                <p>Visualize data with beautiful and informative charts.</p>
            </div>
            <div class="feature">
                <div class="feature-icon">🛠️</div>
                <div class="feature-title">Effortless Preprocessing</div>
                <p>Streamline data preprocessing and preparation.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Get Started Section
    st.markdown("""
        <div class="section-title">🚀 Get Started with AutoEDA</div>
        <p>AutoEDA simplifies your data journey. Upload your dataset and instantly gain insights through automated preprocessing and EDA.</p>
    """, unsafe_allow_html=True)

    # Target Audience
    # st.markdown("""
    #     <div class="section-title">👥 Designed For</div>
    #     <div class="target-audience">
    #         <div class="audience">📊 Data Analysts</div>
    #         <div class="audience">🔎 Data Scientists</div>
    #         <div class="audience">🧐 Business Professionals</div>
    #         <div class="audience">📈 Students & Educators</div>
    #     </div>
    # """, unsafe_allow_html=True)

    # Start Button
    # st.markdown("""
    #     <div style="text-align:center; margin-top:40px;">
    #         <button class="action-button">Start Exploring</button>
    #     </div>
    # """, unsafe_allow_html=True)


def custom_css():
    return """
    <style>
    body {
        background-color: #f9fafc;
        font-family: "Segoe UI", sans-serif;
        color: #333;
    }

    .container {
        text-align: center;
        padding: 40px 0 10px 0;
    }

    .header {
        font-size: 48px;
        font-weight: 700;
        color: #2E86C1;
        margin-bottom: 8px;
    }

    .tagline {
        font-size: 22px;
        color: #555;
    }

    .section-title {
        text-align: center;
        font-size: 28px;
        font-weight: 600;
        color: #1F618D;
        margin: 40px 0 20px 0;
    }

    .features {
        display: flex;
        justify-content: space-evenly;
        flex-wrap: wrap;
        margin-bottom: 40px;
    }

    .feature {
        width: 280px;
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 20px;
        text-align: center;
        transition: transform 0.3s ease-in-out;
        margin: 10px;
    }

    .feature:hover {
        transform: translateY(-6px);
    }

    .feature-icon {
        font-size: 36px;
        margin-bottom: 10px;
    }

    .feature-title {
        font-size: 18px;
        font-weight: bold;
        color: #2874A6;
        margin-bottom: 8px;
    }

    .target-audience {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 16px;
    }

    .audience {
        background: #EAF2F8;
        border-radius: 8px;
        padding: 12px 20px;
        font-weight: 500;
        color: #154360;
        transition: background 0.3s;
    }

    .audience:hover {
        background: #D6EAF8;
    }

    .action-button {
        background-color: #2E86C1;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 28px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.3s;
    }

    .action-button:hover {
        background-color: #1A5276;
    }
    </style>
    """ 
