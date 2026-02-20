import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Trader Analytics & AI", layout="wide")
st.title("🚀 Primetrade.ai - Trader AI & Analytics")

# --- LOAD DATA AND MODELS ---
@st.cache_data
def load_data():
    try:
        df_dash = pd.read_csv('cleaned_dashboard_data.csv')
        df_arch = pd.read_csv('trader_archetypes.csv')
        return df_dash, df_arch
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_model():
    try:
        return joblib.load('profit_predictor_model.pkl')
    except FileNotFoundError:
        return None

df, df_arch = load_data()
model = load_model()

# --- CREATE TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Main Dashboard", "🧩 Behavioral Archetypes (Bonus)", "🔮 Profit Predictor (Bonus)"])

# --- TAB 1: MAIN DASHBOARD ---
with tab1:
    if df is not None:
        st.sidebar.header("Filter Data")
        sentiment_filter = st.sidebar.multiselect("Select Sentiment", options=df['Broad_Sentiment'].unique(), default=['Fear', 'Greed'])
        size_filter = st.sidebar.selectbox("Trade Size Segment", options=['All'] + list(df['size_segment'].unique()))
        
        filtered_df = df[df['Broad_Sentiment'].isin(sentiment_filter)]
        if size_filter != 'All':
            filtered_df = filtered_df[filtered_df['size_segment'] == size_filter]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Traders Analyzed", filtered_df['Account'].nunique())
        col2.metric("Total PnL in Selection", f"${filtered_df['daily_pnl'].sum():,.2f}")
        col3.metric("Average Win Rate", f"{filtered_df['win_rate'].mean():.1%}")

        # ADDED: Download Button for the filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name='filtered_trader_data.csv',
            mime='text/csv',
        )
        st.markdown("---")

        colA, colB = st.columns(2)
        with colA:
            st.subheader("PnL Distribution by Sentiment")
            fig1 = px.box(filtered_df, x="Broad_Sentiment", y="daily_pnl", color="Broad_Sentiment", points=False)
            fig1.update_yaxes(range=[-5000, 5000]) 
            st.plotly_chart(fig1, use_container_width=True)
            
        with colB:
            st.subheader("Trade Frequency by Sentiment")
            fig2 = px.bar(filtered_df.groupby('Broad_Sentiment')['total_trades'].mean().reset_index(), 
                          x='Broad_Sentiment', y='total_trades', color='Broad_Sentiment')
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("Dashboard data not found. Please upload 'cleaned_dashboard_data.csv'.")

# --- TAB 2: ARCHETYPES ---
with tab2:
    st.header("Behavioral Archetypes (K-Means Clustering)")
    
    # ADDED: UI Expander for explanation
    with st.expander("ℹ️ How did we cluster these traders?"):
        st.write("""
        We used a **K-Means Clustering** algorithm. It scales lifetime metrics (Total PnL, Win Rate, Total Trades) 
        and groups traders mathematically into 3 distinct behavioral profiles. 
        * **Conservative Scalpers:** Low size, high frequency, stable but low PnL.
        * **High-Risk Degens:** High frequency, massive PnL swings (boom or bust).
        * **Consistent Whales:** Low frequency, high win rate, high positive PnL.
        """)
    
    if df_arch is not None:
        colA, colB = st.columns(2)
        with colA:
            st.dataframe(df_arch[['Account', 'total_pnl', 'total_trades', 'Archetype']].head(15))
        with colB:
            fig = px.scatter(df_arch, x="total_trades", y="total_pnl", color="Archetype", 
                             title="Trader PnL vs Total Trades")
            st.plotly_chart(fig)
    else:
        st.error("Archetype data not found. Please upload 'trader_archetypes.csv'.")

# --- TAB 3: PREDICTIVE MODEL ---
with tab3:
    st.header("Next-Day Profitability Predictor (Random Forest)")
    st.write("Enter today's metrics to predict if the trader will be profitable tomorrow.")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            trades = st.number_input("Today's Total Trades", min_value=1, value=10)
            pnl = st.number_input("Today's PnL ($)", value=150.0)
        with col2:
            win_rate = st.slider("Today's Win Rate", 0.0, 1.0, 0.5)
            sentiment = st.selectbox("Market Sentiment Today", ["Fear", "Greed"])
            
        submitted = st.form_submit_button("Predict Tomorrow's Performance")
        
        if submitted:
            if model is not None:
                # 0 for Fear, 1 for Greed (assuming standard alphabetical label encoding)
                sentiment_encoded = 1 if sentiment == "Greed" else 0
                
                # Model expects a 2D array, ensuring correct feature order
                features = pd.DataFrame(
                    [[trades, pnl, win_rate, sentiment_encoded]], 
                    columns=['total_trades', 'daily_pnl', 'win_rate', 'Sentiment_Encoded']
                )
                
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0][1]
                
                if prediction == 1:
                    st.success(f"✅ AI Predicts: TRADER WILL PROFIT TOMORROW (Confidence: {probability:.1%})")
                else:
                    st.error(f"⚠️ AI Predicts: TRADER WILL LOSE MONEY TOMORROW (Confidence: {1 - probability:.1%})")
            else:
                st.error("Model file 'profit_predictor_model.pkl' not found.")

    # ADDED: Feature Importance Visualization
    st.markdown("---")
    st.subheader("🧠 How is the AI making this decision?")
    st.write("This chart shows which metrics the Random Forest model relies on most to predict profitability.")
    
    if model is not None:
        # Extract feature importances from the loaded model
        importances = model.feature_importances_
        feature_names = ['Total Trades', 'Daily PnL', 'Win Rate', 'Sentiment']
        
        # Create a dataframe and plot it
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                         color='Importance', color_continuous_scale='Blues')
        fig_imp.update_layout(showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)