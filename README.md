# Automated On-Wafer Parametric Test System

**A detailed hardware related system set up guide has been provided in:**
[system_set_up_guide.pdf](https://github.com/Th3stral/Automated_On-Wafer_Parametric_Test_System/blob/main/system_set_up_guide.pdf)


**If you are only trying to access to the GUI, you could follow the following guidence:**

**Here is a basic guide walk you through the process of installing Python, navigating to your project directory, installing dependencies, and running the test application.**

## Prerequisites
- A computer running Windows (preferred), or Linux (havent't tested)
- Internet connection

## Step 1: Install Python 3.10 , 3.11, or 3.12

1. **Download Python 3.10:**

   - Visit the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Choose the appropriate installer for your operating system (Windows, or Linux).
   - Click on the download link for Python 3.10.

2. **Install Python:**

   - **Windows:**
     - Run the downloaded `.exe` file.
     - Make sure to check the box that says "Add Python 3.10 to PATH" before clicking "Install Now."
   
   - **Linux:**
     - Open a terminal window.
     - Run the following commands:

     ```bash
     sudo apt update
     sudo apt install python3.10
     ```

3. **Verify Installation:**

   Open a terminal or command prompt and run:

   ```bash
   python3.10 --version
   ```

## Step 2: Navigate to Your Project Directory

Open Terminal or Command Prompt:.

Navigate to Your Project Directory:

```bash
cd path/to/your/project/directory
```

Replace `path/to/your/project/directory` with the actual path to your project.

## Step 3: Install Dependencies

Ensure `pip` is Installed:
If `pip` is not installed, you can install it by running:

```bash
python3.10 -m ensurepip --upgrade
```

Install Requirements:
Run the following command to install all necessary packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Step 4: Run the Streamlit App

Use the following command to run the test application:


```bash
streamlit run app.py
```

**Access the App:**
Once the server starts, the default web browser should automatically open and display the application's GUI. 

If it does not, manually open your web browser and navigate to the URL provided by Streamlit in the terminal window, which is typically:

```bash
http://localhost:8501
```

Upon seeing this interface, you are ready to interact with the GUI.

**However, be aware of the instrument-related functionalities; you will first need to complete the “Software Set Up” section in [system_set_up_guide.pdf](https://github.com/Th3stral/Automated_On-Wafer_Parametric_Test_System/blob/main/system_set_up_guide.pdf) to use them.**

![https://imgur.com/SpK1mTd.png](https://imgur.com/SpK1mTd.png)
