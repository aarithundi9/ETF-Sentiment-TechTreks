"""
Tech Treks – ETF Sentiment Analyzer

A Streamlit dashboard that:
1. Takes a user query (news headline, tweet, etc.)
2. Uses LLM + sentiment analysis to pick an ETF and generate sentiment scores
3. Runs XGBoost model with price features + query sentiment
4. Displays 5-day and 1-month predictions

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import sys
import joblib
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the query-based prediction function
from src.models.query_xgboost_model import predict_with_query, load_and_prepare_data

# Page config
st.set_page_config(
    page_title="Tech Treks – ETF Sentiment Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the UI
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a0a0a;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #2a2a4a;
    }
    
    /* ETF button styling */
    .etf-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 10px;
        padding: 15px 25px;
        color: white;
        font-weight: bold;
        margin: 5px;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .etf-button:hover {
        transform: scale(1.05);
    }
    
    .etf-button.selected {
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sentiment bars */
    .sentiment-bar {
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Query input */
    .stTextArea textarea {
        background-color: #1a1a2e;
        border: 2px solid #667eea;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Constants
DATA_DIR = Path("data/processed")
MODEL_DIR = DATA_DIR / "models"

# ETF definitions with keywords for matching
ETF_CONFIG = {
    'QQQ': {
        'name': 'Invesco QQQ Trust',
        'sector': 'Technology',
        'color': '#667eea',
        'keywords': ['tech', 'technology', 'software', 'ai', 'artificial intelligence', 'chip', 'semiconductor', 
                     'nvidia', 'apple', 'microsoft', 'google', 'amazon', 'meta', 'facebook', 'cloud', 'saas',
                     'internet', 'digital', 'cyber', 'data', 'computing', 'nasdaq', 'innovation', 'startup']
    },
    'XLF': {
        'name': 'Financial Select Sector',
        'sector': 'Financials',
        'color': '#f093fb',
        'keywords': ['bank', 'banking', 'finance', 'financial', 'interest rate', 'fed', 'federal reserve',
                     'loan', 'credit', 'mortgage', 'insurance', 'investment', 'wall street', 'jpmorgan',
                     'goldman', 'wells fargo', 'citibank', 'treasury', 'bond', 'yield', 'inflation']
    },
    'XLI': {
        'name': 'Industrial Select Sector', 
        'sector': 'Industrials',
        'color': '#4facfe',
        'keywords': ['industrial', 'manufacturing', 'factory', 'infrastructure', 'construction', 'aerospace',
                     'defense', 'military', 'boeing', 'caterpillar', 'transportation', 'logistics', 'shipping',
                     'railroad', 'machinery', 'equipment', 'engineering', 'automation', 'supply chain']
    },
    'XLY': {
        'name': 'Consumer Discretionary Select',
        'sector': 'Consumer Discretionary', 
        'color': '#00f2fe',
        'keywords': ['consumer', 'retail', 'shopping', 'e-commerce', 'amazon', 'tesla', 'auto', 'car',
                     'electric vehicle', 'ev', 'spending', 'holiday', 'travel', 'hotel', 'restaurant',
                     'entertainment', 'media', 'streaming', 'netflix', 'disney', 'nike', 'luxury']
    }
}


def analyze_query_sentiment(query: str) -> dict:
    """
    Analyze the query to determine sentiment scores.
    
    This is a rule-based implementation. In production, this would call an LLM.
    Your partner can replace this with the actual LLM integration.
    """
    query_lower = query.lower()
    
    # Positive indicators
    positive_words = ['surge', 'soar', 'jump', 'gain', 'rise', 'grow', 'boost', 'record', 'high',
                      'strong', 'beat', 'exceed', 'bullish', 'optimistic', 'recovery', 'boom',
                      'profit', 'success', 'breakthrough', 'innovation', 'demand', 'expansion']
    
    # Negative indicators  
    negative_words = ['crash', 'fall', 'drop', 'decline', 'plunge', 'sink', 'lose', 'loss',
                      'weak', 'miss', 'fail', 'bearish', 'pessimistic', 'recession', 'crisis',
                      'bankruptcy', 'layoff', 'cut', 'warning', 'concern', 'fear', 'risk']
    
    # Count matches
    pos_count = sum(1 for word in positive_words if word in query_lower)
    neg_count = sum(1 for word in negative_words if word in query_lower)
    
    total = pos_count + neg_count + 1  # +1 to avoid division by zero
    
    if pos_count > neg_count:
        positive = 0.5 + (pos_count / total) * 0.4
        negative = 0.1 + (neg_count / total) * 0.2
        score = (positive - negative)
    elif neg_count > pos_count:
        negative = 0.5 + (neg_count / total) * 0.4
        positive = 0.1 + (pos_count / total) * 0.2
        score = (positive - negative)
    else:
        positive = 0.33
        negative = 0.33
        score = 0.0
    
    neutral = max(0.1, 1.0 - positive - negative)
    
    # Normalize to sum to 1
    total_prob = positive + negative + neutral
    positive /= total_prob
    negative /= total_prob
    neutral /= total_prob
    
    return {
        'score': round(score, 3),
        'positive': round(positive, 3),
        'negative': round(negative, 3),
        'neutral': round(neutral, 3)
    }


def select_etf_from_query(query: str) -> str:
    """
    Select the most relevant ETF based on query keywords.
    
    This is a rule-based implementation. In production, this would use an LLM.
    """
    query_lower = query.lower()
    
    scores = {}
    for ticker, config in ETF_CONFIG.items():
        score = sum(1 for keyword in config['keywords'] if keyword in query_lower)
        scores[ticker] = score
    
    # If no matches, default to QQQ (tech is most common)
    if max(scores.values()) == 0:
        return 'QQQ'
    
    return max(scores, key=scores.get)


@st.cache_data
def load_etf_data(ticker: str) -> pd.DataFrame:
    """Load ETF data from CSV."""
    filepath = DATA_DIR / f"{ticker}total.csv"
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


def create_price_chart(df: pd.DataFrame, ticker: str, prediction_5d: str, prediction_1m: str) -> go.Figure:
    """Create a clean price line chart."""
    df_recent = df.tail(90).copy()
    
    # Determine colors based on predictions
    color_5d = '#00ff88' if prediction_5d == 'UP' else '#ff4444'
    color_1m = '#00ff88' if prediction_1m == 'UP' else '#ff4444'
    
    fig = go.Figure()
    
    # Main price line
    fig.add_trace(go.Scatter(
        x=df_recent['date'],
        y=df_recent['close'],
        mode='lines',
        name=ticker,
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    # Add prediction arrows/annotations
    latest_date = df_recent['date'].iloc[-1]
    latest_price = df_recent['close'].iloc[-1]
    
    # 5-day prediction point
    pred_5d_date = latest_date + timedelta(days=7)
    pred_5d_price = latest_price * (1.02 if prediction_5d == 'UP' else 0.98)
    
    # 1-month prediction point
    pred_1m_date = latest_date + timedelta(days=30)
    pred_1m_price = latest_price * (1.05 if prediction_1m == 'UP' else 0.95)
    
    # Prediction line
    fig.add_trace(go.Scatter(
        x=[latest_date, pred_5d_date, pred_1m_date],
        y=[latest_price, pred_5d_price, pred_1m_price],
        mode='lines+markers',
        name='Prediction',
        line=dict(color=color_1m, width=2, dash='dash'),
        marker=dict(size=10, symbol='triangle-up' if prediction_1m == 'UP' else 'triangle-down')
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=True),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig


def create_sentiment_gauge(positive: float, negative: float, neutral: float) -> go.Figure:
    """Create a horizontal stacked bar for sentiment breakdown."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=['Sentiment'],
        x=[positive],
        name='Positive',
        orientation='h',
        marker_color='#00ff88',
        text=f'{positive:.0%}',
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        y=['Sentiment'],
        x=[neutral],
        name='Neutral',
        orientation='h', 
        marker_color='#888888',
        text=f'{neutral:.0%}',
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        y=['Sentiment'],
        x=[negative],
        name='Negative',
        orientation='h',
        marker_color='#ff4444',
        text=f'{negative:.0%}',
        textposition='inside'
    ))
    
    fig.update_layout(
        barmode='stack',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Title
    st.markdown('<h1 class="main-title">🚀 Tech Treks – ETF Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888;">Enter a news headline or query to analyze its impact on ETFs</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'selected_etf' not in st.session_state:
        st.session_state.selected_etf = None
    if 'query_sentiment' not in st.session_state:
        st.session_state.query_sentiment = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    # Main layout: 3 columns
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    # =================================
    # LEFT COLUMN: Query Input & ETF Selection
    # =================================
    with col_left:
        st.markdown("### 📝 Enter Query")
        
        query = st.text_area(
            "News headline or query:",
            placeholder="e.g., 'NVIDIA reports record AI chip sales amid surging demand'",
            height=100,
            label_visibility="collapsed"
        )
        
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 ETF Selection")
        
        # ETF buttons
        for ticker, config in ETF_CONFIG.items():
            is_selected = st.session_state.selected_etf == ticker
            button_style = "primary" if is_selected else "secondary"
            
            if st.button(
                f"**{ticker}** - {config['sector']}", 
                key=f"etf_{ticker}",
                use_container_width=True,
                type=button_style
            ):
                st.session_state.selected_etf = ticker
                st.rerun()
        
        # Show selected ETF info
        if st.session_state.selected_etf:
            etf = st.session_state.selected_etf
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); 
                        padding: 15px; border-radius: 10px; margin-top: 10px;'>
                <h4 style='margin: 0; color: {ETF_CONFIG[etf]["color"]};'>{etf}</h4>
                <p style='margin: 5px 0 0 0; color: #888; font-size: 0.9em;'>
                    {ETF_CONFIG[etf]['name']}<br>
                    Sector: {ETF_CONFIG[etf]['sector']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # =================================
    # Process query when analyze button is clicked
    # =================================
    if analyze_button and query:
        # Select ETF based on query
        selected_etf = select_etf_from_query(query)
        st.session_state.selected_etf = selected_etf
        
        # Analyze sentiment
        sentiment = analyze_query_sentiment(query)
        st.session_state.query_sentiment = sentiment
        
        # Get prediction from model
        result = predict_with_query(selected_etf, sentiment)
        st.session_state.prediction_result = result
        
        st.rerun()
    
    # =================================
    # CENTER COLUMN: Main Analysis Panel
    # =================================
    with col_center:
        st.markdown("### 📈 ETF Sentiment Analyzer")
        
        if st.session_state.prediction_result and st.session_state.selected_etf:
            result = st.session_state.prediction_result
            etf = st.session_state.selected_etf
            
            # Load ETF data for chart
            df = load_etf_data(etf)
            
            # Get predictions
            pred_5d = result['predictions']['5d']
            pred_1m = result['predictions']['1m']
            
            # Metrics row
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric(
                    "Selected ETF",
                    etf,
                    ETF_CONFIG[etf]['sector']
                )
            
            with metric_cols[1]:
                st.metric(
                    "Current Price",
                    f"${result['latest_price']:.2f}",
                    ""
                )
            
            with metric_cols[2]:
                icon_5d = "🟢" if pred_5d['direction'] == 'UP' else "🔴"
                pct_5d = (pred_5d['up_probability'] - 0.5) * 10
                st.metric(
                    "5-Day Forecast",
                    f"{icon_5d} {pred_5d['direction']}",
                    f"{pct_5d:+.1f}% expected"
                )
            
            with metric_cols[3]:
                icon_1m = "🟢" if pred_1m['direction'] == 'UP' else "🔴"
                pct_1m = (pred_1m['up_probability'] - 0.5) * 20
                st.metric(
                    "1-Month Forecast",
                    f"{icon_1m} {pred_1m['direction']}",
                    f"{pct_1m:+.1f}% expected"
                )
            
            # Price chart
            if df is not None:
                chart = create_price_chart(df, etf, pred_5d['direction'], pred_1m['direction'])
                st.plotly_chart(chart, use_container_width=True)
            
            # Confidence bars
            st.markdown("#### Prediction Confidence")
            conf_cols = st.columns(2)
            
            with conf_cols[0]:
                st.markdown(f"**5-Day:** {pred_5d['up_probability']:.1%} P(UP)")
                st.progress(pred_5d['up_probability'])
            
            with conf_cols[1]:
                st.markdown(f"**1-Month:** {pred_1m['up_probability']:.1%} P(UP)")
                st.progress(pred_1m['up_probability'])
        
        else:
            # Empty state
            st.markdown("""
            <div style='text-align: center; padding: 50px; color: #666;'>
                <h2>👆 Enter a query to get started</h2>
                <p>Type a news headline, tweet, or any text related to markets.</p>
                <p>The analyzer will:</p>
                <ol style='text-align: left; display: inline-block;'>
                    <li>Select the most relevant ETF</li>
                    <li>Analyze sentiment impact</li>
                    <li>Predict 5-day and 1-month price direction</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    # =================================
    # RIGHT COLUMN: Sentiment Breakdown
    # =================================
    with col_right:
        st.markdown("### 🎯 Sentiment Analysis")
        
        if st.session_state.query_sentiment:
            sentiment = st.session_state.query_sentiment
            
            # Overall sentiment score
            score = sentiment['score']
            score_color = '#00ff88' if score > 0 else '#ff4444' if score < 0 else '#888888'
            score_label = 'Positive' if score > 0.1 else 'Negative' if score < -0.1 else 'Neutral'
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); 
                        padding: 20px; border-radius: 15px; text-align: center;'>
                <h1 style='font-size: 3rem; margin: 0; color: {score_color};'>{score:+.2f}</h1>
                <p style='color: #888; margin: 5px 0 0 0;'>Overall Score</p>
                <p style='color: {score_color}; font-weight: bold; margin: 10px 0 0 0;'>{score_label}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Sentiment breakdown
            st.markdown("#### Breakdown")
            
            # Positive
            st.markdown(f"""
            <div style='margin: 10px 0;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #00ff88;'>🟢 Positive</span>
                    <span style='color: #00ff88;'>{sentiment['positive']:.1%}</span>
                </div>
                <div style='background: #1a1a1a; border-radius: 5px; height: 10px; margin-top: 5px;'>
                    <div style='background: #00ff88; width: {sentiment['positive']*100}%; height: 100%; border-radius: 5px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Neutral
            st.markdown(f"""
            <div style='margin: 10px 0;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #888888;'>⚪ Neutral</span>
                    <span style='color: #888888;'>{sentiment['neutral']:.1%}</span>
                </div>
                <div style='background: #1a1a1a; border-radius: 5px; height: 10px; margin-top: 5px;'>
                    <div style='background: #888888; width: {sentiment['neutral']*100}%; height: 100%; border-radius: 5px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Negative
            st.markdown(f"""
            <div style='margin: 10px 0;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #ff4444;'>🔴 Negative</span>
                    <span style='color: #ff4444;'>{sentiment['negative']:.1%}</span>
                </div>
                <div style='background: #1a1a1a; border-radius: 5px; height: 10px; margin-top: 5px;'>
                    <div style='background: #ff4444; width: {sentiment['negative']*100}%; height: 100%; border-radius: 5px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Stacked bar visualization
            st.markdown("---")
            gauge = create_sentiment_gauge(
                sentiment['positive'],
                sentiment['negative'], 
                sentiment['neutral']
            )
            st.plotly_chart(gauge, use_container_width=True)
            
        else:
            st.markdown("""
            <div style='text-align: center; padding: 30px; color: #666;'>
                <p>Sentiment scores will appear here after analysis</p>
            </div>
            """, unsafe_allow_html=True)
    
    # =================================
    # Footer
    # =================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        <p>Tech Treks ETF Analyzer | XGBoost Model with 24 Features (20 Technical + 4 Sentiment)</p>
        <p>Data: Yahoo Finance | Not financial advice</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
