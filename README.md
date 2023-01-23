#### About

# Complaint Loss Detector (CLD)

Still under construction ...

An NLP and ML powered application for monetary loss reduction

## **Problem**

Each year, financial institutions have lost millions of dollars due to their customers' complaints. Many complaints are lengthy texts and hard to identify key reasons for filing complaints against those institutions in addition to its large quantity. It is impossible for a financial institutions to conduct thorough analyses and summarization of those complaints just by its customer service team and response the complaints in a timely fashion.

In addition, as more and more products the financial institutions serve, the variety of the complaints becomes even more difficult for identifying the key reasons of customer complaints. As results, the customer service team might miss some of the key reasons for customers' complaints and opportunities to address the complaints with appropriate customer service experts in a correct way, which leads to monetary loss to the complaints.

## **Insights**

## **Data**

![](https://www.consumerfinance.gov/static/img/logo_237x50.c7c2ba6c929f.png)

CFPB provides a glimpse of [consumers' complaints](https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data) against a variety of financial institutions. Its database contains around 3.1M complaints, and many of those have caused monetary loss of the institutions.

The complaints includes:
* 1.42M complaints were filed against three major credit reporting agencies
    * Only tiny number of them (< 0.1%) were result of monetary loss
* 670K complaints have consumers' complaint narratives against the other financial institutions.

## **Solutions**
Transform the problem into a unsupervised and supervised ML problem:   

* variable: 'Consumer complaint narrative'
* target: 'Company response to consumer'   

## **Tools and Techs**

* Python
* Data Cleaning + EDA
    * Jupyter
    * Pandas
    * Seaborn
    * Feature Engineering
    * Geopy
    * Folium
* NLP + Modeling
    * NLTK
    * spaCy
    * Scikit-learn
    * Worldcloud
* Deployment
    * Flask
    * Streamlit
    * Dash
    * uWSGI
    * AWS EC2

## **Deployment**  
Click the [link](https://bit.ly/mld_dashboard) for the 1st iteration of the app as shown below. You can use the `test.csv` file to play with it.  
   
**CAUTION**: the cheap AWS EC2 is not powerful enough to utilize the 520MB model file, please be patient.   

  <img src="image_02.png" width=500>
