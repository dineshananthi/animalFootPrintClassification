import requests
import streamlit as st

from core.vistra_afp_classification import ProjectException
from main import APP_TITLE

api_endpoint = "http://127.0.0.1:8000/animal-footprint-classification"

# Define correct username and password
CORRECT_USERNAME = "admin"
CORRECT_PASSWORD = "password@123"


def main():
    # Check if user is logged in
    if not st.session_state.get('logged_in', False):
        login_page()
    else:
        image_classification_page()


def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Check if login button is clicked
    if st.button("Login"):
        # Validate username and password
        if username == CORRECT_USERNAME and password == CORRECT_PASSWORD:
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password. Please try again.")
        # Set background image for login page
        st.markdown(
            f"""
                 <style>
                 .stApp {{
                     background: url("https://getwallpapers.com/wallpaper/full/1/8/9/942406-most-popular-wildlife-backgrounds-for-desktop-1920x1080-for-full-hd.jpg");
                     background-size: cover
                 }}
                 </style>
                 """,
            unsafe_allow_html=True
        )


def image_classification_page():
    st.title(f"{APP_TITLE}")
    st.markdown(
        f"""
                     <style>
                     .stApp {{
                         background: url("https://w0.peakpx.com/wallpaper/979/131/HD-wallpaper-giraffes-evening-sunset-africa-wildlife-african-animals.jpg");
                         background-size: cover
                     }}
                     </style>
                     """,
        unsafe_allow_html=True
    )

    # Set Streamlit options to adjust the layout
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showImageFormat', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    uploaded_file = st.file_uploader("Upload an image for animal foot print classification task",
                                     type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_contents = uploaded_file.read()
        st.image(file_contents, caption="Uploaded Image", use_column_width=True)

        if st.button('Classify', key="classify_button"):
            call_api(file_contents)
        if st.button("Back", key="back_button"):
            st.session_state.logged_in = False
            st.session_state.logged_in = True
        if st.button("Logout", key="logout_button"):
            st.session_state.logged_in = False
            st.session_state.logged_in = False


def call_api(image_bytes):
    try:
        files = {'image_path': image_bytes}
        response = requests.post(api_endpoint, files=files)

        if response.status_code == 200:
            response_json = response.json()
            st.write("Image classification successful!")
            st.write(response_json)
            # Create pie chart
            st.title("Confidence Score Pie Chart")
            st.write("Class Label:", response_json["predictedLabel"])
            # Plot pie chart
            chart_data = {"Confidence Score": response_json["confidenceScore"]}
            st.write(chart_data)
            st.write("Confidence Chart:")
            st.bar_chart(chart_data, width=400, height=300)
            return True
        else:
            st.write(f"API call failed with status code: {response.status_code}")
            raise ProjectException(response.text)
    except Exception as ex:
        print(ex)


main()
