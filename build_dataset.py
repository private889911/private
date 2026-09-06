import pandas as pd
import requests
import time
import urllib.parse
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# loading both lists
INPUT_FILE = '100 AI & Non-AI List [Pre-Sentiment & Price].xlsx'
OUTPUT_FILE = 'Master_Dataset_Analyzed.csv'
MAX_REVIEWS = 300  

analyzer = SentimentIntensityAnalyzer()

# implementing regex
PATTERN_AI = re.compile(r'\b(ai|artificial intelligence|chatgpt|midjourney|stable diffusion|llm|genai|generative|ai slop|machine learning)\b', re.IGNORECASE)
PATTERN_SCAM = re.compile(r'\b(scam|cash grab|cashgrab|asset flip|rip off|ripoff|abandonware|abandoned|grift|money grab|fraud)\b', re.IGNORECASE)
PATTERN_TECH = re.compile(r'\b(bug|bugs|buggy|glitch|glitches|glitchy|crash|crashes|crashing|unplayable|stutter|optimization|fps|broken)\b', re.IGNORECASE)

def get_game_sentiment(appid):
    """gets reviews, runs vader, flags themes"""
    reviews_analyzed = 0
    compound_scores = []
    
    # themes
    ai_mentions = 0
    scam_mentions = 0
    tech_mentions = 0
    
    cursor = '*'
    print(f"  -> Extracting AppID: {appid}...", end=" ")
    
    while reviews_analyzed < MAX_REVIEWS:
        safe_cursor = urllib.parse.quote(cursor)
        url = f"https://store.steampowered.com/appreviews/{appid}?json=1&language=english&filter=recent&num_per_page=100&cursor={safe_cursor}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"[API ERROR {response.status_code}]", end=" ")
                break
                
            data = response.json()
            if data.get('success') != 1 or not data.get('reviews'):
                break 
                
            for review in data['reviews']:
                text = review.get('review', '')
                if text.strip(): 
                    # vader score
                    score = analyzer.polarity_scores(text)['compound']
                    compound_scores.append(score)
                    
                    # theme flags
                    if PATTERN_AI.search(text): ai_mentions += 1
                    if PATTERN_SCAM.search(text): scam_mentions += 1
                    if PATTERN_TECH.search(text): tech_mentions += 1
                    
                    reviews_analyzed += 1
                    if reviews_analyzed >= MAX_REVIEWS:
                        break
                        
            new_cursor = data.get('cursor')
            if not new_cursor or new_cursor == cursor:
                break
            cursor = new_cursor
            
            time.sleep(0.5) # rate limit pause
            
        except Exception as e:
            print(f"[TIMEOUT/ERROR]", end=" ")
            break

    if not compound_scores:
        print("[0 REVIEWS FOUND]")
        return None, 0, None, None, None, None
        
    mean_sentiment = sum(compound_scores) / len(compound_scores)
    positive_ratio = sum(1 for s in compound_scores if s > 0.05) / len(compound_scores)
    
    # calc theme mention pct
    ai_ratio = ai_mentions / reviews_analyzed
    scam_ratio = scam_mentions / reviews_analyzed
    tech_ratio = tech_mentions / reviews_analyzed
    
    print(f"[DONE: {reviews_analyzed} revs | Mean: {mean_sentiment:.3f} | AI Mentions: {ai_mentions}]")
    return mean_sentiment, reviews_analyzed, positive_ratio, ai_ratio, scam_ratio, tech_ratio

def process_cohort(sheet_name, is_ai_cohort):
    print(f"\n{'='*50}\nPROCESSING COHORT: {sheet_name}\n{'='*50}")
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name).iloc[1:].copy()
    df = df.dropna(subset=['Game Title'])
    
    # add new stat columns
    df['Mean_Sentiment_Score'] = None
    df['Reviews_Analyzed'] = 0
    df['Positive_Sentiment_Ratio'] = None
    df['Theme_AI_Ratio'] = None
    df['Theme_Scam_Ratio'] = None
    df['Theme_Tech_Ratio'] = None
    df['Is_AI_Cohort'] = 1 if is_ai_cohort else 0
    
    for index, row in df.iterrows():
        appid = str(row['AppID']).replace('.0', '').strip()
        if appid == 'nan' or not appid:
            continue
            
        mean_sent, count, pos_ratio, ai_r, scam_r, tech_r = get_game_sentiment(appid)
        
        df.at[index, 'Mean_Sentiment_Score'] = mean_sent
        df.at[index, 'Reviews_Analyzed'] = count
        df.at[index, 'Positive_Sentiment_Ratio'] = pos_ratio
        df.at[index, 'Theme_AI_Ratio'] = ai_r
        df.at[index, 'Theme_Scam_Ratio'] = scam_r
        df.at[index, 'Theme_Tech_Ratio'] = tech_r
        
        time.sleep(1.5) 
        
    return df

if __name__ == "__main__":
    start_time = time.time()
    
    df_ai = process_cohort('AI', is_ai_cohort=True)
    df_non_ai = process_cohort('Non-AI Control', is_ai_cohort=False)
    
    master_df = pd.concat([df_ai, df_non_ai], ignore_index=True)
    master_df.to_csv(OUTPUT_FILE, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*50}\nEXTRACTION COMPLETE IN {elapsed:.2f} MINUTES.\nMaster dataset saved to: {OUTPUT_FILE}\nTotal valid records: {len(master_df)}\n{'='*50}")