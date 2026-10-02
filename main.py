from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import traceback

# Global variables to cache dataset in memory
dataset = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event to load and preprocess datasets into memory only once when the server boots.
    """
    print("Initializing Board Game AI Backend...")
    try:
        games_df = pd.read_csv('games.csv')
        ratings_df = pd.read_csv('ratings.csv')
        
        # Standardize columns
        games_df.columns = [str(c).lower().strip() for c in games_df.columns]
        ratings_df.columns = [str(c).lower().strip() for c in ratings_df.columns]
        
        games_rename = {}
        for col in games_df.columns:
            if col in ['id', 'game_id', 'gameid']: games_rename[col] = 'game_id'
            if col in ['name', 'title']: games_rename[col] = 'name'
            if 'min' in col and 'play' in col: games_rename[col] = 'min_players'
            if 'max' in col and 'play' in col: games_rename[col] = 'max_players'
            if col in ['image', 'thumbnail', 'img']: games_rename[col] = 'image'
        games_df = games_df.rename(columns=games_rename)
        games_df = games_df.loc[:, ~games_df.columns.duplicated()]
        
        ratings_rename = {}
        for col in ratings_df.columns:
            if col in ['user', 'username', 'user_id']: ratings_rename[col] = 'user_id'
            if col in ['id', 'game_id', 'gameid']: ratings_rename[col] = 'game_id'
            if col in ['rating', 'score']: ratings_rename[col] = 'rating'
        ratings_df = ratings_df.rename(columns=ratings_rename)
        ratings_df = ratings_df.loc[:, ~ratings_df.columns.duplicated()]
        
        # Fallback for missing image column
        if 'image' not in games_df.columns:
            games_df['image'] = 'https://via.placeholder.com/400x400/1e293b/ffffff?text=No+Cover'
        else:
            games_df['image'] = games_df['image'].fillna('https://via.placeholder.com/400x400/1e293b/ffffff?text=No+Cover')
            
        print(f"Loaded {len(games_df)} games and {len(ratings_df)} ratings.")
        
        # DENSE SAMPLING FIX: User-Centric Sampling
        # To support a dynamic search bar, we anchor our user base around the top 1000 most popular games.
        if len(ratings_df) > 50000:
            print("Optimizing matrix density for User Profiles...")
            top_popular_games = ratings_df['game_id'].value_counts().nlargest(1000).index
            
            essential_users_df = ratings_df[ratings_df['game_id'].isin(top_popular_games)]
            essential_user_ids = essential_users_df['user_id'].unique()
            
            # Limit to 2,000 users to prevent MemoryError when pivoting
            if len(essential_user_ids) > 2000:
                np.random.seed(42)
                essential_user_ids = np.random.choice(essential_user_ids, 2000, replace=False)
                
            # Extract ALL ratings made by these 2,000 users (dense profile)
            ratings_df = ratings_df[ratings_df['user_id'].isin(essential_user_ids)]
            
    except FileNotFoundError:
        print("WARNING: Dataset CSVs not found. Using dummy dataset for demonstration.")
        
        games_df = pd.DataFrame({
            'game_id': range(1, 101),
            'name': [f'Dummy Board Game {i}' for i in range(1, 101)],
            'min_players': np.random.randint(1, 4, 100),
            'max_players': np.random.randint(4, 10, 100),
            'image': [f'https://via.placeholder.com/400x400/1e293b/ffffff?text=Game+{i}'] * 100
        })
        
        users = [f'User_{i}' for i in np.random.randint(1, 2000, 100000)]
        games = np.random.choice(games_df['game_id'].values, 100000)
        ratings = np.random.randint(1, 11, 100000)
        ratings_df = pd.DataFrame({'user_id': users, 'game_id': games, 'rating': ratings}).drop_duplicates(subset=['user_id', 'game_id'])

    # Compute Top 20 Popular Games based on ratings count
    game_counts = ratings_df['game_id'].value_counts()
    top_20_ids = game_counts.nlargest(20).index
    valid_top_20_ids = [gid for gid in top_20_ids if gid in games_df['game_id'].values]
    
    # Retrieve game info and preserve popularity order
    top_20_games = games_df.set_index('game_id').loc[valid_top_20_ids].reset_index()
    dataset['popular'] = top_20_games[['game_id', 'name', 'image']].to_dict(orient='records')

    # Compute Thai Popular Games
    thai_ids = [128882, 147949, 131357, 148228, 39856, 175549, 171228, 188834, 156129, 710]
    valid_thai_ids = [gid for gid in thai_ids if gid in games_df['game_id'].values]
    thai_games = games_df.set_index('game_id').loc[valid_thai_ids].reset_index()
    dataset['popular_thai'] = thai_games[['game_id', 'name', 'image']].to_dict(orient='records')

    dataset['games'] = games_df
    dataset['ratings'] = ratings_df
    print("Backend Ready.")
    yield
    dataset.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RatingInput(BaseModel):
    game_id: int
    rating: float

class RecommendRequest(BaseModel):
    player_count: int
    ratings: List[RatingInput]

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found in root directory.</h1>"

@app.get("/popular")
async def get_popular_games():
    """Returns the top 20 most popular games."""
    return dataset.get('popular', [])

@app.get("/popular-thai")
async def get_popular_thai_games():
    """Returns the top 10 most popular games in Thai cafes."""
    return dataset.get('popular_thai', [])

@app.get("/search")
async def search_games(q: str = Query(..., min_length=1)):
    """
    Dynamically search for board games based on query. Returns top 10 matches.
    """
    try:
        games_df = dataset['games']
        
        # Case-insensitive substring match
        matches = games_df[games_df['name'].str.contains(q, case=False, na=False)]
        
        # Sort by popularity if available
        sort_col = next((c for c in matches.columns if 'usersrated' in c or 'owned' in c), None)
        if sort_col:
            matches = matches.sort_values(by=sort_col, ascending=False)
            
        top_10 = matches.head(10)
        results = []
        for row in top_10.itertuples():
            results.append({
                "game_id": row.game_id,
                "name": str(getattr(row, 'name', 'Unknown')),
                "image": str(getattr(row, 'image', 'https://via.placeholder.com/400x400/1e293b/ffffff?text=No+Cover'))
            })
        return results
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(traceback.format_exc()))

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    """
    Executes User-User Collaborative Filtering (Cosine Similarity).
    """
    try:
        games_df = dataset['games']
        ratings_df = dataset['ratings']
        
        user_ratings = {r.game_id: r.rating for r in req.ratings}
        if not user_ratings:
            raise HTTPException(status_code=400, detail="Must rate at least one game to compute similarity.")
            
        # 1. Hard Filter by Player Count
        valid_games = games_df[(games_df['min_players'] <= req.player_count) & (games_df['max_players'] >= req.player_count)]
        valid_game_ids = valid_games['game_id'].unique()
        
        # 2. Build Utility Matrix
        relevant_game_ids = set(valid_game_ids).union(user_ratings.keys())
        filtered_ratings = ratings_df[ratings_df['game_id'].isin(relevant_game_ids)]
        
        utility_matrix = filtered_ratings.pivot_table(index='user_id', columns='game_id', values='rating', aggfunc='mean')
        
        # 3. Append Active User
        active_user_id = 'ACTIVE_USER'
        active_user_series = pd.Series(user_ratings, name=active_user_id)
        utility_matrix = pd.concat([utility_matrix, active_user_series.to_frame().T])
        
        # 4. Fill NaNs with 0 (Standard Cosine Similarity to avoid zero-variance collapse)
        centered_matrix = utility_matrix.fillna(0).astype(float)
        
        # 5. Compute Cosine Similarity (Vectorized NumPy)
        active_vector = centered_matrix.loc[active_user_id].values
        other_users_matrix = centered_matrix.drop(active_user_id)
        other_vectors = other_users_matrix.values
        
        dot_products = np.dot(other_vectors, active_vector)
        norm_active = np.linalg.norm(active_vector)
        norms_others = np.linalg.norm(other_vectors, axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            similarities = dot_products / (norm_active * norms_others)
            similarities = np.nan_to_num(similarities, nan=0.0)
            
        sim_series = pd.Series(similarities, index=other_users_matrix.index)
        
        # 6. Find Neighborhood N
        top_5_users = sim_series[sim_series > 0].nlargest(5)
        
        if top_5_users.empty:
            raise HTTPException(status_code=404, detail="Neighborhood is empty. Try rating more games or different games.")
            
        # 7. Predict Unplayed Games
        unplayed_valid_games = [gid for gid in valid_game_ids if gid not in user_ratings]
        predictions = []
        
        for gid in unplayed_valid_games:
            if gid not in utility_matrix.columns:
                continue
            top_5_ratings = utility_matrix.loc[top_5_users.index, gid].dropna()
            if top_5_ratings.empty:
                continue
                
            valid_sims = top_5_users.loc[top_5_ratings.index]
            predicted_rating = np.dot(valid_sims, top_5_ratings) / valid_sims.sum()
            predictions.append({'game_id': gid, 'predicted_rating': predicted_rating})
            
        if not predictions:
            raise HTTPException(status_code=404, detail="Insufficient neighbor data to predict valid games.")
            
        # 8. Extract Top 3 and Return
        predictions_df = pd.DataFrame(predictions).nlargest(3, 'predicted_rating')
        
        recommendations_out = []
        for row in predictions_df.itertuples():
            g_info = games_df[games_df['game_id'] == row.game_id].iloc[0]
            recommendations_out.append({
                "name": str(g_info['name']),
                "predicted_rating": float(round(row.predicted_rating, 2)),
                "image": str(g_info['image'])
            })
            
        proof_out = [{"user_id": str(uid), "similarity_score": float(round(sim, 4))} for uid, sim in top_5_users.items()]
        
        return {
            "recommendations": recommendations_out,
            "proof": proof_out
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print("CRITICAL CRASH:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(traceback.format_exc()))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
