import streamlit as st
import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
def clean_skills_column(text):
    if pd.isna(text):
        return ''
    return str(text).lower().replace('[^a-zA-Z0-9, ]', '', regex=True)

@st.cache_resource 
def retrieve_model(filename='job_model.pkl'):
    try:
        model_artifacts = joblib.load(filename)
        vectorizer = model_artifacts.get('vectorizer')
        kmeans_model = model_artifacts.get('kmeans_model')
        if vectorizer is None or kmeans_model is None:
            st.error(f"Error: 'vectorizer' or 'kmeans_model' not found in {filename}.")
            return None, None
        return vectorizer, kmeans_model
    except FileNotFoundError:
        st.error(f"Error: Model file '{filename}' not found. Please ensure it's in the same directory as app.py.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None


def fetch_jobs_from_karkidi(search_term="data science", max_pages=1):
    st.info(f"Fetching jobs for '{search_term}' across {max_pages} pages...")
    jobs_data = []
    base_url = "https://www.karkidi.com/jobs"

    for page in range(1, max_pages + 1):
        url = f"{base_url}?search={search_term}&page={page}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            job_listings = soup.find_all('div', class_='job-listing-card')
            if not job_listings:
                st.warning(f"No job listings found on page {page}.")
                break

            for job in job_listings:
                title = job.find('h2', class_='job-title').text.strip() if job.find('h2', class_='job-title') else 'N/A'
                company = job.find('p', class_='company-name').text.strip() if job.find('p', class_='company-name') else 'N/A'
                location = job.find('span', class_='location').text.strip() if job.find('span', class_='location') else 'N/A'
                experience = job.find('span', class_='experience').text.strip() if job.find('span', class_='experience') else 'N/A'
                skills_div = job.find('div', class_='skills')
                skills = ", ".join([skill.text.strip() for skill in skills_div.find_all('span', class_='skill-tag')]) if skills_div else 'N/A'
                summary = job.find('div', class_='job-summary').text.strip() if job.find('div', class_='job-summary') else 'N/A'

                jobs_data.append({
                    'Title': title,
                    'Company': company,
                    'Location': location,
                    'Experience': experience,
                    'Key Skills': skills,
                    'Summary': summary
                })
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching page {page}: {e}")
            break
    return pd.DataFrame(jobs_data)



st.set_page_config(layout="wide")
st.title("Karkidi Job Skill Cluster Predictor")
st.markdown("Enter job skills or scrape jobs to predict which cluster they belong to, using your pre-trained model.")


vectorizer, kmeans_model = retrieve_model()

if vectorizer is None or kmeans_model is None:
    st.error("Model could not be loaded. Please check 'job_model.pkl' file.")
else:
    st.success("Clustering model loaded successfully!")

    st.header("Predict Cluster for Custom Skills")
    custom_skills_input = st.text_area(
        "Enter job skills (comma-separated or free text):",
        "Python, Machine Learning, Data Analysis, SQL, Cloud, AWS"
    )

    if st.button("Predict Cluster for Custom Skills"):
        if vectorizer and kmeans_model and custom_skills_input:
            cleaned_skills = clean_skills_column(custom_skills_input)
            if cleaned_skills:
                
                try:
                    input_vector = vectorizer.transform([cleaned_skills])
                    
                    predicted_cluster = kmeans_model.predict(input_vector)[0]
                    st.success(f"The entered skills belong to Cluster: **{predicted_cluster}**")
                except Exception as e:
                    st.error(f"Error during prediction: {e}. Ensure input format is consistent.")
            else:
                st.warning("Please enter some skills to predict.")
        else:
            st.warning("Model not loaded or no skills entered.")

    st.markdown("---")

   
    st.header("Scrape Jobs and Predict Clusters")
    st.sidebar.header("Scraping Parameters (Optional)")
    scrape_search_term = st.sidebar.text_input("Search term for scraping", "data science")
    scrape_max_pages = st.sidebar.slider("Number of pages to scrape", 1, 5, 1)

    if st.sidebar.button("Scrape and Predict Clusters"):
        if scrape_search_term:
            with st.spinner(f"Fetching jobs for '{scrape_search_term}' and predicting clusters..."):
                scraped_jobs_df = fetch_jobs_from_karkidi(scrape_search_term, scrape_max_pages)
                if not scraped_jobs_df.empty:
                    
                    scraped_jobs_df['Cleaned Skills'] = scraped_jobs_df['Key Skills'].apply(clean_skills_column)

                    
                    df_for_prediction = scraped_jobs_df[scraped_jobs_df['Cleaned Skills'].str.strip() != ''].copy()

                    if not df_for_prediction.empty:
                       
                        try:
                            job_vectors = vectorizer.transform(df_for_prediction['Cleaned Skills'])
                            df_for_prediction['Predicted Cluster'] = kmeans_model.predict(job_vectors)

                            
                            final_display_df = scraped_jobs_df.merge(
                                df_for_prediction[['Title', 'Predicted Cluster']],
                                on='Title',
                                how='left'
                            )
                            final_display_df['Predicted Cluster'] = final_display_df['Predicted Cluster'].fillna('N/A (No Skills)').astype(str)

                            st.subheader("Scraped Jobs with Predicted Clusters:")
                            st.dataframe(final_display_df)

                            
                            st.subheader("Distribution of Predicted Clusters:")
                            cluster_counts = final_display_df['Predicted Cluster'].value_counts().sort_index()
                            st.bar_chart(cluster_counts)

                        except Exception as e:
                            st.error(f"Error during bulk prediction: {e}")
                    else:
                        st.warning("No valid job skills found after scraping for prediction.")
                else:
                    st.warning("No jobs were scraped. Check your search term or the website.")
        else:
            st.sidebar.warning("Please enter a search term for scraping.")

st.markdown("---")
st.markdown("Developed for the Karkidi Job Scraper and Clustering Project.")
