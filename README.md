# Customer-Segmentation-using-K-Means-Clustering
Description Customer segmentation is a technique used by businesses to divide customers into groups with similar characteristics. In this project, K-Means Clustering is applied to identify patterns in customer data such as annual income and spending score.

Project Name:-
Customer Segmentation using K-Means Clustering

Problem Statement:-
Businesses often have many customers with different income levels and spending habits. Treating all customers the same can reduce marketing effectiveness and waste resources.
The goal of this project is to use Machine Learning (K-Means Clustering) to divide customers into meaningful groups based on their Annual Income and Spending Score. These customer groups help businesses create targeted marketing strategies, improve customer satisfaction, and increase sales.


Technologies Used
Python
Pandas – data handling
NumPy – numerical operations
Matplotlib – data visualization
Seaborn – advanced visualization
Scikit-learn – machine learning (K-Means Clustering)
Dataset

This project uses the Mall Customers Dataset.

Features:
CustomerID – Unique customer ID
Gender – Male / Female
Age – Customer age
Annual Income (k$) – Annual income in thousand dollars
Spending Score (1-100) – Score based on spending behavior

The dataset is commonly used for beginner machine learning and clustering projects.



Step 1: Install Required Libraries

Run the following command in terminal:
pip install -r requirements.txt

Step 2: Dataset

The dataset Mall_Customers.csv is located in the data/ folder.

Step 3: Run Python File

Run your Python script:
python customer_segmentation.py

Step 4: Outputs Generated

The program will:

Load dataset
Preprocess data
Show customer distribution graph
Apply Elbow Method and save elbow.png in images/
Create clusters using K-Means
Show final cluster graph with centroids and save clusters.png in images/
Generate cluster summary

Results

Key Outcomes:
Successfully grouped customers into 5 different clusters
Identified customer behavior patterns
Generated business insights from each segment
Visualized clusters using scatter plots and centroids

Example Customer Segments:
High Income + High Spending → Premium Customers
High Income + Low Spending → Careful Customers
Low Income + High Spending → Impulsive Buyers
Low Income + Low Spending → Budget Customers
Medium Income + Medium Spending → Regular Customers

Business Value:
Better targeted marketing
Improved customer retention
Personalized offers
Increased sales opportunities







