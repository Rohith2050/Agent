import pandas as pd
import sqlite3
import requests
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import tempfile
import json

st.set_page_config(page_title="📈 Campaign Advisor", layout="wide")
st.title("📊 Interactive Campaign Performance Analyzer")

OPENROUTER_API_KEY = "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"
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def retrieve_campaign_data(sheet_url=None, db_connection=None, api_url=None):
    if sheet_url:
        try:
            df = pd.read_csv(sheet_url)
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            df.fillna(0, inplace=True)
            return df
        except Exception as e:
            st.error(f"❌ Failed to load sheet: {e}")
            return None
    elif db_connection:
        query = "SELECT * FROM campaigns"
        try:
            df = pd.read_sql(query, db_connection)
            return df
        except Exception as e:
            st.error(f"❌ Failed to load data from database: {e}")
            return None
    elif api_url:
        try:
            response = requests.get(api_url)
            data = response.json()
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            st.error(f"❌ Failed to load data from API: {e}")
            return None
    else:
        st.error("❌ No valid data source provided.")
        return None
        
def get_refined_query(historical_changes, df):
    prompt = f"""
    Based on the historical campaign performance trends, how should I adjust Budget, Bids, Targeting, and Creative
    to increase ROAS? Consider the following previous adjustments:

    {historical_changes}

    Also, consider the trends in ROAS, Amount Spent, Impressions, and other relevant metrics from the dataset below:
    {df[['campaign', 'roas', 'amount_spent', 'impressions', 'reach', 'cpm']].tail(5).to_dict(orient='records')}

    Please provide your suggestions in JSON format, including fields: 'budget', 'bids', 'targeting', 'creative', and 'explanation'.
    """
    return prompt

def tool_selection(query, df, db_connection=None, api_url=None):
    if "average" in query or "mean" in query:  
        print("Using Pandas to process the data")
        return df.describe()  
    elif "campaign" in query: 
        print("Using SQL to query the database")
        return retrieve_campaign_data(db_connection=db_connection)
    elif "data" in query:
        print("Using API to fetch external data")
        return retrieve_campaign_data(api_url=api_url)
    else:
        print("Query not recognized, returning dataset for manual inspection")
        return df

def get_campaign_recommendations(historical_changes, retriever, df):
    refined_query = get_refined_query(historical_changes, df)
    response = qa_chain.run({"question": refined_query})
    
    return response

historical_changes = []

def apply_changes(budget_adjustment, bid_adjustment, targeting, creative):
    change = {
        'budget': budget_adjustment,
        'bids': bid_adjustment,
        'targeting': targeting,
        'creative': creative
    }
    historical_changes.append(change)
    return change


sheet_url = st.text_input("📥 Paste your Google Sheet CSV export link", placeholder="https://docs.google.com/spreadsheets/d/e/your-sheet/pub?output=csv")
df = None

if sheet_url:
    df = retrieve_campaign_data(sheet_url=sheet_url)

if df is not None:
    st.success("✅ Data loaded successfully!")
    st.dataframe(df, use_container_width=True)
    with st.spinner("🔄 Embedding campaign data..."):
        documents = []
        for _, row in df.iterrows():
            content = f"""
Campaign: {row['campaign']}
Ad Set: {row['ad_set']}
Ad Creative: {row['ad_creative']}
ROAS: {row['roas']}
Amount Spent: ${row['amount_spent']}
Impressions: {row['impressions']}
Outbound Clicks: {row['outbound_clicks']}
Purchases ROAS: {row['purchase_roas']}
Reach: {row['reach']}
Cost per Purchase: ${row['cost_per_purchase']}
"""
            doc = Document(
                page_content=content.strip(),
                metadata={
                    "campaign": row["campaign"],
                    "ad_set": row["ad_set"],
                    "ad_creative": row["ad_creative"],
                    "age": row["age"],
                    "gender": row["gender"],
                    "bid_strategy": row["bid_strategy"],
                    "budget": row["budget"],
                    "attribution_window": row["attribution_window"],
                    "frequency": row["frequency"],
                    "cpm": row["cpm"]
                }
            )
            documents.append(doc)

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embedding_model)

        index_path = tempfile.mkdtemp()
        vectorstore.save_local(index_path)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    st.success("✅ Vector index ready!")
    
    llm = ChatOpenAI(
        model="mistralai/mixtral-8x7b",  
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    
st.subheader("⚙️ Optimization Suggestion")
if st.button("📈 Generate Suggestions"):
    with st.spinner("🔬 Generating suggestions..."):
        optimization_response = get_campaign_recommendations(historical_changes, retriever, df)
        st.markdown("### 📊 Suggested Strategy")
        st.code(optimization_response, language="json")

st.subheader("🔧 Adjust Campaign Parameters")
budget = st.slider("Budget Adjustment (%)", -50, 50, 0)
bids = st.slider("Bids Adjustment (%)", -50, 50, 0)
targeting = st.text_input("Targeting Adjustments", "")
creative = st.text_input("Creative Adjustments", "")

if st.button("💾 Apply Changes"):
    change = apply_changes(budget, bids, targeting, creative)
    st.success("Changes applied successfully! Displaying updated performance metrics...")

st.subheader("💬 Ask a Campaign Question")
user_question = st.text_input("Type your question", placeholder="Which ad set had the best ROAS among males aged 25-34?")

if user_question:
    with st.spinner("🔍 Analyzing..."):
        # Use the ConversationalRetrievalChain to answer the question
        response = qa_chain.run({"question": user_question})
        
        st.markdown("### 🤖 Answer")
        st.write(response)
