# Collaborative Filtering Recommendation System with TensorFlow and FastAPI
Copyright (c) 2024 Brian Scanlon

This project is a recommendation system built using collaborative filtering and matrix factorization. It generates personalized product recommendations based on user browsing history and reviews. The system is built with TensorFlow for model training and FastAPI for serving recommendations as an API. It is designed to work with a .csv file as a data source.

## Features

- Collaborative Filtering using TensorFlow's Embedding layers
- FastAPI endpoints for retrieving recommendations
- Evaluation metrics using Mean Absolute Error (MAE)

## Prerequisites

- Python 3.8 or later

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/recommendation-system.git
   cd recommendation-system
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure to have the `sample_training_data.csv` file in the same directory as `app_with_csv.py`. This CSV file should contain user interaction data in the format:

   ```csv
   user_id,item_id,browsing_history,rating
   1,101,"101,102,103",5
   2,102,"101,104,105",3
   ```

   A sample CSV file can be generated using the provided sample.

## Usage

1. Run the FastAPI server:

   ```bash
   python app_with_csv.py
   ```

2. The API will be available at `http://127.0.0.1:8000`.

### API Endpoints

- **Get Recommendations for a User**

  Retrieves product recommendations for a given user.

  ```http
  GET /recommendations/{user_id}
  ```

- **Get Similar Items to a Product**

  Retrieves items similar to a specified product.

  ```http
  GET /similar_items/{item_id}
  ```

## Testing

To test the recommendation model, the system calculates the Mean Absolute Error (MAE) upon loading the model. This provides an initial evaluation metric for the recommendation quality.

## Notes

- TensorFlow may require additional setup based on your system configuration, particularly if you wish to use GPU acceleration.
- This app can be integrated with a Node.js backend and a React frontend to display recommendations on an e-commerce website.

## Next Steps

Potential improvements and next steps:

- Integrate with a larger dataset or database for production.
- Experiment with hybrid models that combine collaborative filtering with content-based filtering.
