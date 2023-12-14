import streamlit as st

def documentation():
    st.title('DOCUMENTATION')

    # Introduction
    st.title("Python for Data Analysis - Landslide Prediction")

    st.markdown("""
    Landslides are natural disasters with global consequences, impacting lives and properties. Predicting and understanding their impact is crucial for effective disaster management. This project utilizes Python for data analysis, specifically focusing on machine learning techniques to analyze the Global Landslide Catalog Export dataset provided by NASA. The goal is to develop a predictive model that anticipates the number of fatalities resulting from landslides, aiding in early warning systems and disaster management efforts.
    """)

    # Global Landslide Catalog
    st.header("Global Landslide Catalog")

    st.markdown("""
    The dataset originates from NASA's Global Landslide Catalog (GLC), documenting rainfall-triggered landslide events worldwide. It is a comprehensive amalgamation of data from media outlets, disaster databases, scientific publications, and reliable resources. Key features include information about the event, location, landslide characteristics, and more.
    """)


    # Dataset Features
    st.subheader("Dataset Features")

    dataset_features = [
        "source_name: News entity reporting the event",
        "source_link: Website URL of the news source",
        "event_id: Identification number for the event",
        "event_date: Date of the landslide event",
        "event_time: Time of the landslide event",
        "event_title: Title of the news story",
        "event_description: Description of the event",
        "location_description: Description of the event's location",
        "landslide_category: Categorization of the landslide",
        "landslide_trigger: Cause of the landslide",
        "landslide_size: Size classification of the landslide",
        "landslide_setting: Environment where the landslide occurred",
        "fatality_count: Number of fatalities",
        "injury_count: Number of injuries",
        "storm_name: Associated storm name (if applicable)",
        "photo_link: URL link to related photos",
        "notes: Additional notes about the event",
        "country_name: Name of the country where the event occurred",
        "country_code: Code representing the country",
        "admin_division_name: Administrative division name",
        "admin_division_population: Population of the administrative division",
        "longitude: Longitude coordinate of the event location",
        "latitude: Latitude coordinate of the event location",
        "submitted_date: Date of submission of the event information",
        "created_date: Date of dataset entry creation",
        "last_edited_date: Date of the last edit made to the entry"
    ]

    for feature in dataset_features:
        st.write(feature)

    # Project Structure Overview
    st.header('Project Structure Overview')

    # Folder: app
    st.subheader('1. app:')
    st.write("""
        - __init__.py: Initialization file.
        - project_structure: Description or documentation of the project's structure.
        - app.py: Primary file creating the application.
        - data_analysis.py: File dedicated to data analysis for the application.
        - documentation.py: File handling documentation aspects of the application.
        - homepage.py: File concerning the application's homepage.
        - map.py: File managing map-related functionalities within the application.
        - ml_models.py: File incorporating machine learning models for the application.
    """)

    # Folder: data
    st.subheader('2. data:')
    st.write("""
        - Contains the crucial Global Landslide Catalog (GLC) CSV file, the primary dataset for analysis.
    """)

    # Folder: model
    st.subheader('3. model:')
    st.write("""
        - Includes the model.py file, housing code related to machine learning model creation and training.
    """)

    # Folder: src/img
    st.subheader('4. src/img:')
    st.write("""
        - Stores images utilized within the project.
    """)

    # Files
    st.header('Files:')
    st.write("""
        - .gitignore: Specifies intentionally untracked files to be ignored by Git.
        - README.md: Main documentation or guide for users/contributors.
        - exploration.ipynb: Jupyter Notebook showcasing exploration activities.
        - landslide_prediction.ipynb: Jupyter Notebook dedicated to landslide prediction, possibly showcasing model implementation and evaluation.
        - requirements.txt: Specifies necessary dependencies or packages for the project.
    """)
