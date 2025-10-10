# Winnipeg_AVM
This is my submission for the technical test for Opportunity No. 44161. 

# Automated Valuation Model (AVM)

## Description
This repository provides a Python implementation of an Automated Valuation Model (AVM) for predicting the assessed value of residential properties in Winnipeg. It uses open source libraries and fetches data live from the City of Winnipeg Open Data Portal.

## Repository Structure
- `avm_main.py`: Main Python script containing data loading, exploratory analysis, feature engineering, preprocessing, model training, and evaluation.
- `requirements.txt`: Lists all Python package dependencies with fixed versions.
- `README.md`: This documentation file.
- `Output` : Contains the log file and plots
- `Documentation`: Contains two documentation files explaining the project:
- - accessibility_documentation : For non-technical audiences
- - technical_documentation : For technical audiences

## How to run the Python script:
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/winnipeg-avm.git
   cd winnipeg-avm
   ```

2. Create and activate a virtual environment (Python 3.8+):
   ```bash
   python -m venv venv
   source venv/bin/activate   
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure you are connected to the internet (for data fetch).
2. Run the main script:
   ```bash
   python avm_main.py
   ```
3. Logs will be stored in `winnipeg_avm.log` in the project root.

## Data
- Data is fetched from the OData endpoint:
  ```text
  https://data.winnipeg.ca/api/odata/v4/d4mq-wa44
  ```
- No local data files are included; the script handles fetching and initial cleaning.

## Logging
- The script logs progress and metrics to `avm_main.log`.

## Assumptions
- Python 3.8 or higher is used.
- A stable internet connection is available.
- Dataset schema remains consistent.

## Contact
For issues or questions, please open an issue on the repository.

