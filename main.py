import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu
import data_analysis_functions as function
import data_preprocessing_function as preprocessing_function
import home_page
import base64

# --- Page Configuration ---
st.set_page_config(page_icon="✨", page_title="AutoEDA", layout="wide")

# --- Hide Streamlit’s Default Elements ---
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- Sidebar Section ---
with st.sidebar:
    
    # Add animated cart icon 
    st.markdown("""
        <style>
        @keyframes float {
            0% { transform: translatey(0px); }
            50% { transform: translatey(-10px); }
            100% { transform: translatey(0px); }
        }
        .animated-icon {
            width: 80px;
            margin: 0 auto;
            display: block;
            animation: float 3s ease-in-out infinite;
        }
        </style>
        <img src="https://cdn-icons-png.flaticon.com/512/8146/8146003.png" class="animated-icon" />
    """, unsafe_allow_html=True)

    st.title("AutoEDA ✨")

    st.markdown("#### Automated Exploratory Data Analysis & Processing")

    uploaded_file = st.file_uploader("📤 Upload CSV or Excel File", type=["csv", "xls"])
    use_example_data = st.checkbox("Use Example Titanic Dataset", value=False)

    # st.markdown("---")
    # st.markdown("### Connect with Me:")
    # st.markdown(
    #     """
    #     [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devang-chavan/)
    #     [![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Devang-C)
    #     """
    # )

# --- Navbar (Horizontal Menu) ---
selected = option_menu(
    menu_title=None,
    options=['Home', 'Data Exploration', 'Data Preprocessing'],
    icons=['house-heart', 'bar-chart-fill', 'hammer'],
    orientation='horizontal',
    styles={
        "container": {"padding": "0!important", "background-color": "#F0F2F6"},
        "icon": {"color": "#5A5DF0", "font-size": "20px"},
        "nav-link": {
            "font-size": "17px",
            "text-align": "center",
            "margin": "5px",
            "--hover-color": "#EEE",
        },
        "nav-link-selected": {"background-color": "#5A5DF0", "color": "white"},
    },
)

# --- Home Page ---
if selected == 'Home':
    home_page.show_home_page()

# --- Load Dataset ---
if uploaded_file:
    df = function.load_data(uploaded_file)
    if 'new_df' not in st.session_state:
        st.session_state.new_df = df.copy()
elif use_example_data:
    df = function.load_data(file="example_dataset/titanic.csv")
    if 'new_df' not in st.session_state:
        st.session_state.new_df = df
else:
    df = None

# --- Empty State ---
if df is None and selected != 'Home':
    st.markdown("### 📂 Please upload a dataset or use the sample Titanic dataset from the sidebar.")
else:
    # ----------------- DATA EXPLORATION -----------------
    if selected == 'Data Exploration':
        tab1, tab2 = st.tabs(["📊 Dataset Overview", "🔍 Data Visualization"])
        num_columns, cat_columns = function.categorical_numerical(df)

        with tab1:
            st.header("Dataset Overview")
            st.info("Quick summary and structure of your dataset.")
            function.display_dataset_overview(df, cat_columns, num_columns)

            st.subheader("Missing Values")
            function.display_missing_values(df)

            st.subheader("Statistics & Visualization")
            function.display_statistics_visualization(df, cat_columns, num_columns)

            st.subheader("Data Types")
            function.display_data_types(df)

            function.search_column(df)

        with tab2:
            st.header("Data Visualization")
            function.display_individual_feature_distribution(df, num_columns)

            st.subheader("Scatter Plot")
            function.display_scatter_plot_of_two_numeric_features(df, num_columns)

            if cat_columns:
                st.subheader("Categorical Variable Analysis")
                function.categorical_variable_analysis(df, cat_columns)
            else:
                st.info("No categorical columns found.")

            st.subheader("Numerical Feature Exploration")
            if num_columns:
                function.feature_exploration_numerical_variables(df, num_columns)
            else:
                st.warning("No numerical columns to explore.")

            st.subheader("Categorical vs Numerical Analysis")
            if cat_columns and num_columns:
                function.categorical_numerical_variable_analysis(df, cat_columns, num_columns)
            else:
                st.info("Analysis unavailable due to missing categorical/numerical data.")

    # ----------------- DATA PREPROCESSING -----------------
    if selected == 'Data Preprocessing':
        st.header("🛠 Data Preprocessing Dashboard")

        if st.button("🔄 Revert to Original Dataset"):
            st.session_state.new_df = df.copy()

        # --- Column Removal ---
        st.subheader("🧹 Remove Unwanted Columns")
        columns_to_remove = st.multiselect("Select Columns to Remove", st.session_state.new_df.columns)
        if st.button("Remove Selected Columns"):
            st.session_state.new_df = preprocessing_function.remove_selected_columns(st.session_state.new_df, columns_to_remove)
            st.success("Selected columns removed successfully.")
        st.dataframe(st.session_state.new_df)

        # --- Handle Missing Data ---
        st.subheader("🧩 Handle Missing Data")
        missing_count = st.session_state.new_df.isnull().sum()
        if missing_count.any():
            selected_option = st.selectbox("Choose Method", ["Remove Rows", "Fill Missing Values"])
            if selected_option == "Remove Rows":
                cols = st.multiselect("Select Columns to Remove Missing Rows", st.session_state.new_df.columns)
                if st.button("Apply Removal"):
                    st.session_state.new_df = preprocessing_function.remove_rows_with_missing_data(st.session_state.new_df, cols)
                    st.success("Missing rows removed.")
            else:
                cols = st.multiselect("Select Numeric Columns", st.session_state.new_df.select_dtypes(include=['number']).columns)
                method = st.selectbox("Fill Method", ["mean", "median", "mode"])
                if st.button("Apply Filling"):
                    st.session_state.new_df = preprocessing_function.fill_missing_data(st.session_state.new_df, cols, method)
                    st.success(f"Missing data filled using {method}.")
            function.display_missing_values(st.session_state.new_df)
        else:
            st.info("✅ No missing values found.")

        # --- Encoding ---
        st.subheader("🧠 Encode Categorical Data")
        cat_cols = st.session_state.new_df.select_dtypes(include=['object']).columns
        if not cat_cols.empty:
            selected_cols = st.multiselect("Select Columns", cat_cols)
            method = st.selectbox("Encoding Method", ['One Hot Encoding', 'Label Encoding'])
            if st.button("Apply Encoding"):
                if method == "One Hot Encoding":
                    st.session_state.new_df = preprocessing_function.one_hot_encode(st.session_state.new_df, selected_cols)
                else:
                    st.session_state.new_df = preprocessing_function.label_encode(st.session_state.new_df, selected_cols)
                st.success("Encoding Applied Successfully.")
        else:
            st.info("No categorical columns found.")

        # --- Scaling ---
        st.subheader("📏 Feature Scaling")
        num_cols = st.session_state.new_df.select_dtypes(include=['number']).columns
        selected_cols = st.multiselect("Select Columns for Scaling", num_cols)
        method = st.selectbox("Scaling Method", ["Standardization", "Min-Max Scaling"])
        if st.button("Apply Scaling"):
            if method == "Standardization":
                st.session_state.new_df = preprocessing_function.standard_scale(st.session_state.new_df, selected_cols)
            else:
                st.session_state.new_df = preprocessing_function.min_max_scale(st.session_state.new_df, selected_cols)
            st.success(f"{method} Applied Successfully.")
        st.dataframe(st.session_state.new_df)

        # --- Outlier Handling ---
        st.subheader("📈 Outlier Detection & Handling")
        selected_col = st.selectbox("Select Column", num_cols)
        fig, ax = plt.subplots()
        sns.boxplot(data=st.session_state.new_df, x=selected_col, ax=ax)
        st.pyplot(fig)

        outliers = preprocessing_function.detect_outliers_zscore(st.session_state.new_df, selected_col)
        if outliers:
            st.warning("Outliers Detected!")
            st.write(outliers)
        else:
            st.info("No outliers found.")
        method = st.selectbox("Outlier Handling Method", ["Remove", "Transform"])
        if st.button("Apply Outlier Handling"):
            if method == "Remove":
                st.session_state.new_df = preprocessing_function.remove_outliers(st.session_state.new_df, selected_col, outliers)
            else:
                st.session_state.new_df = preprocessing_function.transform_outliers(st.session_state.new_df, selected_col, outliers)
            st.success("Outlier handling completed.")
        st.dataframe(st.session_state.new_df)

        # --- Download Preprocessed Data ---
        if st.session_state.new_df is not None:
            csv = st.session_state.new_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.download_button("⬇️ Download Preprocessed Data", data=csv, file_name="preprocessed_data.csv", mime="text/csv")
