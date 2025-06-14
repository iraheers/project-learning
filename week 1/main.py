from fastapi import FastAPI, HTTPException
import sqlite3
from pathlib import Path
app = FastAPI()

DB_PATH = Path(__file__).parent / "nutrition.db"

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/food/{food_name}")
def get_nutrition(food_name: str):
    """Endpoint to get nutrition data for a specific food"""
    conn = None
    try:
        # Connect to database
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Query database
        cursor.execute('''
        SELECT name, calories, protein, carbs, fat, serving_size 
        FROM foods WHERE LOWER(name)=?
        ''', (food_name.lower(),))
        
        result = cursor.fetchone()
        
        if result:
            return {
                "name": result[0],
                "calories": result[1],
                "protein": result[2],
                "carbs": result[3],
                "fat": result[4],
                "serving_size": result[5]
            }
        raise HTTPException(status_code=404, detail="Food not found")
        
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()  # Ensure connection is always closed

@app.get("/foods/")
def list_foods():
    """Endpoint to list all foods in database"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT name, calories FROM foods")
        return {
            "foods": [
                {"name": row[0], "calories": row[1]}
                for row in cursor.fetchall()
            ]
        }
    finally:
        if conn:
            conn.close()