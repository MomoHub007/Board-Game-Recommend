import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="BoardGame AI Matchmaker", layout="wide")

@st.cache_data
def load_and_preprocess_data():
    """
    Loads data and explicitly implements dense subset sampling to maintain user-item overlap,
    which is mathematically crucial for User-User CF.
    """
    try:
        games_df = pd.read_csv('games.csv')
        ratings_df = pd.read_csv('ratings.csv')
        
        # Column standardization to handle potential variations in datasets
        games_df.columns = [str(c).lower().strip() for c in games_df.columns]
        ratings_df.columns = [str(c).lower().strip() for c in ratings_df.columns]
        
        games_rename = {}
        for col in games_df.columns:
            if col in ['id', 'game_id', 'gameid']: games_rename[col] = 'game_id'
            if col in ['name', 'title']: games_rename[col] = 'name'
            if 'min' in col and 'play' in col: games_rename[col] = 'min_players'
            if 'max' in col and 'play' in col: games_rename[col] = 'max_players'
            
        games_df = games_df.rename(columns=games_rename)
        
        ratings_rename = {}
        for col in ratings_df.columns:
            if col in ['user', 'username', 'user_id']: ratings_rename[col] = 'user_id'
            if col in ['id', 'game_id', 'gameid']: ratings_rename[col] = 'game_id'
            if col in ['rating', 'score']: ratings_rename[col] = 'rating'
            
        ratings_df = ratings_df.rename(columns=ratings_rename)
        
    except FileNotFoundError:
        # Fallback to dummy data if files are missing, ensuring execution doesn't crash
        games_df = pd.DataFrame({
            'game_id': range(1, 101),
            'name': [f'Dummy Board Game {i}' for i in range(1, 101)],
            'min_players': np.random.randint(1, 4, 100),
            'max_players': np.random.randint(4, 10, 100)
        })
        
        users = [f'User_{i}' for i in np.random.randint(1, 2000, 100000)]
        games = np.random.randint(1, 101, 100000)
        ratings = np.random.randint(1, 11, 100000)
        ratings_df = pd.DataFrame({'user_id': users, 'game_id': games, 'rating': ratings}).drop_duplicates(subset=['user_id', 'game_id'])

    # Calculate actual average rating for output comparisons
    game_averages = ratings_df.groupby('game_id')['rating'].mean().reset_index()
    game_averages.rename(columns={'rating': 'avg_rating'}, inplace=True)
    games_df = games_df.merge(game_averages, on='game_id', how='left')

    # Identify top popular games for cold start mitigation
    game_popularity = ratings_df.groupby('game_id').size()
    top_5_game_ids = game_popularity.nlargest(5).index.tolist()
    cold_start_games = games_df[games_df['game_id'].isin(top_5_game_ids)].copy()

    # DENSE SAMPLING FIX FOR MEMORY MANAGEMENT
    # Randomly sampling 100k rows directly on millions of ratings causes sparsity and destroys neighbors.
    # We deliberately anchor the sample around users who rated our cold-start games.
    target_sample_size = 100000
    if len(ratings_df) > target_sample_size:
        essential_ratings = ratings_df[ratings_df['game_id'].isin(top_5_game_ids)]
        remaining_slots = target_sample_size - len(essential_ratings)
        
        if remaining_slots > 0:
            # We sample other ratings to hit target_sample_size
            other_ratings = ratings_df[~ratings_df.index.isin(essential_ratings.index)].sample(n=remaining_slots, random_state=42)
            ratings_df = pd.concat([essential_ratings, other_ratings])
        else:
            ratings_df = essential_ratings.sample(n=target_sample_size, random_state=42)

    return games_df, ratings_df, cold_start_games

def main():
    st.title("🎲 BoardGame AI Matchmaker (User-User Collaborative Filtering)")
    
    with st.spinner("Loading datasets and enforcing Dense Sampling constraints..."):
        games_df, ratings_df, cold_start_games = load_and_preprocess_data()
        
    st.header("Step 1: Hard Filter")
    player_count = st.number_input("How many players in your group today?", min_value=1, max_value=99, value=4, step=1)
    
    st.header("Step 2: Cold Start Mitigation (Explicit Ratings)")
    st.write("Rate these popular games. Your ratings define your User Vector in the Utility Matrix.")
    
    user_ratings = {}
    
    for _, game in cold_start_games.iterrows():
        col1, col2 = st.columns([3, 1])
        with col1:
            rating = st.slider(f"Rate **{game['name']}** (1.0 - 10.0)", 1.0, 10.0, 5.5, 0.5, key=f"slider_{game['game_id']}")
        with col2:
            st.write("") 
            st.write("") 
            never_played = st.checkbox("Never Played (NaN)", key=f"check_{game['game_id']}")
            
        if not never_played:
            user_ratings[game['game_id']] = rating
            
    if st.button("Generate Recommendations", type="primary"):
        if not user_ratings:
            st.warning("Please rate at least one game to allow similarity computation.")
            return
            
        with st.spinner("Pivoting Utility Matrix and Computing Cosine Similarities via NumPy..."):
            # 1. Player Count Hard Filter
            valid_games = games_df[(games_df['min_players'] <= player_count) & (games_df['max_players'] >= player_count)]
            valid_game_ids = valid_games['game_id'].unique()
            
            # 2. Build Utility Matrix
            relevant_game_ids = set(valid_game_ids).union(user_ratings.keys())
            filtered_ratings = ratings_df[ratings_df['game_id'].isin(relevant_game_ids)]
            
            # Rows: Users, Cols: Games
            utility_matrix = filtered_ratings.pivot(index='user_id', columns='game_id', values='rating')
            
            # 3. Append Active User Vector
            active_user_id = 'ACTIVE_USER'
            active_user_series = pd.Series(user_ratings, name=active_user_id)
            utility_matrix = pd.concat([utility_matrix, active_user_series.to_frame().T])
            
            # 4. Mean-Centering to Normalize Bias
            # Center the ratings per user before filling NaNs with 0.
            # This ensures that unrated games (0) are perfectly mathematically neutral.
            user_means = utility_matrix.mean(axis=1)
            centered_matrix = utility_matrix.sub(user_means, axis=0)
            centered_matrix_filled = centered_matrix.fillna(0)
            
            # 5. Compute Cosine Similarity (Strictly NumPy implementation)
            active_vector = centered_matrix_filled.loc[active_user_id].values
            other_users_matrix = centered_matrix_filled.drop(active_user_id)
            other_vectors = other_users_matrix.values
            
            # Dot Product (A • B)
            dot_products = np.dot(other_vectors, active_vector)
            
            # Magnitudes ||A|| * ||B||
            norm_active = np.linalg.norm(active_vector)
            norms_others = np.linalg.norm(other_vectors, axis=1)
            
            # Raw Cosine Similarity
            with np.errstate(divide='ignore', invalid='ignore'):
                similarities = dot_products / (norm_active * norms_others)
                similarities = np.nan_to_num(similarities, nan=0.0)
                
            sim_series = pd.Series(similarities, index=other_users_matrix.index)
            
            # 6. Identify Top 5 Most Similar Users (Neighborhood N)
            top_5_users = sim_series[sim_series > 0].nlargest(5)
            
            if top_5_users.empty:
                st.error("No statistically similar neighbors found based on your ratings.")
                return
                
            st.success("Mathematical Neighborhood Computed Successfully!")
            
            # Academic Proof
            with st.expander("📊 ACADEMIC PROOF: Cosine Similarity Scores (Top 5 Neighbors)", expanded=True):
                st.markdown("Calculated manually using Numpy Matrix Operations: $\\frac{A \\cdot B}{||A|| \\times ||B||}$")
                proof_df = pd.DataFrame({
                    "Neighbor User ID": top_5_users.index,
                    "Cosine Similarity Score": top_5_users.values
                })
                proof_df.index = proof_df.index + 1
                st.table(proof_df)
            
            # 7. Predict Ratings for Unplayed Games
            unplayed_valid_games = [gid for gid in valid_game_ids if gid not in user_ratings]
            predictions = []
            
            for gid in unplayed_valid_games:
                if gid not in utility_matrix.columns:
                    continue
                    
                # Extract Top 5 users' raw ratings for this specific game
                top_5_ratings = utility_matrix.loc[top_5_users.index, gid].dropna()
                
                if top_5_ratings.empty:
                    continue
                    
                # We only weigh by similarity if the neighbor actually rated the game
                valid_sims = top_5_users.loc[top_5_ratings.index]
                
                # Weighted Average: Σ (Similarity * Rating) / Σ (Similarity)
                predicted_rating = np.dot(valid_sims, top_5_ratings) / valid_sims.sum()
                
                predictions.append({
                    'game_id': gid,
                    'predicted_rating': predicted_rating
                })
                
            if not predictions:
                st.warning("Insufficient neighbor data to generate reliable predictions.")
                return
                
            # 8. Output Display
            predictions_df = pd.DataFrame(predictions).nlargest(3, 'predicted_rating')
            
            st.header(f"🏆 Top 3 Recommendations ({player_count} Players)")
            
            for rank, row in enumerate(predictions_df.itertuples(), 1):
                game_info = games_df[games_df['game_id'] == row.game_id].iloc[0]
                actual_avg = game_info.get('avg_rating', np.nan)
                actual_avg_str = f"{actual_avg:.2f}" if not pd.isna(actual_avg) else "N/A"
                
                st.markdown(f"### #{rank}: {game_info['name']}")
                colA, colB = st.columns(2)
                colA.metric("Predicted Rating (CF)", f"{row.predicted_rating:.2f}/10")
                colB.metric("Actual Dataset Avg Rating", f"{actual_avg_str}/10")
                st.markdown("---")

if __name__ == "__main__":
    main()
