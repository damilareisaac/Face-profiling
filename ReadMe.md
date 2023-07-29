This guide will walk you through the steps to setup Virtual environments and run
the code for this application

## Prerequisites

Make sure you have Python installed on your system. You can download the latest Python version from the official website: https://www.python.org/downloads/

## Step 1: Clone or Download the Project

Clone or download the project from the repository or obtain the project files in any other way you prefer.

## Step 2: Set Up Virtual Environment

#### Windows:

1. Open the command prompt (CMD) or PowerShell.
2. Navigate to the project directory using the cd command.
3. Create a virtual environment with the following command:
   `python -m venv venv`
4. Activate the virtual environment: `venv\Scripts\activate`

#### Linux and macOS:

1. Open a terminal.
2. Navigate to the project directory using the cd command.
3. Create a virtual environment with the following command:
   `python3 -m venv venv`
   or using python if python3 is set as the default version:
   `python -m venv venv`

4. Activate the virtual environment:
   `source venv/bin/activate`

## Step 3: Install Requirements

With the virtual environment active, install the required packages from the requirements.txt file. Ensure you are in the project directory and the virtual environment is activated. Run the following command:
`pip install -r requirements.txt`

## Step 4: Running the Project

Now that the virtual environment is set up and the requirements are installed, you can run your Python project as follows:
In the project folder, from your terminal

1. To use webcam option, run:
   `python main.py`
   **make sure that neccessary permissions are granted to allow the app use your camera**

2. To run the analyis on specific picture file, put to picture in the folder `images_db` and run:
   `python main.py --input image -i images_db/NAME_OF_IMAGE`
