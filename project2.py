import sys
sys.path.insert(0, 'chromedriver')
import re
import os
import json
import numpy as np
import requests
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, QuickReply, QuickReplyButton, MessageAction
)
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from bs4 import BeautifulSoup
from selenium import webdriver
import chromedriver_autoinstaller
from selenium.webdriver.common.by import By
from datetime import datetime

# Initialize SentenceTransformer model
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# Neo4j connection details
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "Password"

# Initialize Neo4j driver for managing chat history
class ChatHistory:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()
    def store_chat_history(self, user_id, question, answer):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.driver.session() as session:
            session.run('''
                MERGE (u:User {userID: $user_id})
                CREATE (q:Question {text: $question, timestamp: $timestamp})-[:ASKED_BY]->(u)
                CREATE (a:Answer {text: $answer, timestamp: $timestamp})-[:ANSWERED]->(q)
            ''', user_id=user_id, question=question, answer=answer, timestamp=timestamp)

    def check_chat_history(self, user_id, question):
        with self.driver.session() as session:
            result = session.run('''
                MATCH (u:User {userID: $user_id})-[:ASKED_BY]->(q:Question {text: $question})-[:ANSWERED]->(a:Answer)
                RETURN a.text AS answer
                LIMIT 1
            ''', user_id=user_id, question=question)
            record = result.single()
            if record:
                return record["answer"]  # Return the stored answer if it exists
            return None
# Initialize the Neo4j chat history manager
chat_history = ChatHistory(neo4j_uri, neo4j_user, neo4j_password)

# Cache for categories and dishes
categories_cache = None
dish_info_cache = {}

# Function to scrape recipe categories from Krua.co
def scrape_categories():
    global categories_cache
    if categories_cache is not None:
        return categories_cache

    url = "https://krua.co/recipe"
    
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    chromedriver_autoinstaller.install()

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    driver.implicitly_wait(5)
    html = driver.page_source
    mysoup = BeautifulSoup(html, "html.parser")

    # Extract categories
    select_element = mysoup.find("select", class_="chakra-select css-3d59fr")
    options = select_element.find_all("option")

    categories = {option.text: option['value'] for option in options if option['value']}
    
    categories_cache = categories
    return categories

# Function to compute similarity between sentences
def compute_similar(corpus, sentence):
    a_vec = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
    b_vec = model.encode([sentence], convert_to_tensor=True, normalize_embeddings=True)
    similarities = np.inner(b_vec.cpu().numpy(), a_vec.cpu().numpy()).flatten()
    return similarities

# Function to get user details from Neo4j
def get_user_name(user_id):
    with chat_history.driver.session() as session:
        result = session.run("MATCH (u:User {userID: $user_id}) RETURN u.name AS name", user_id=user_id)
        record = result.single()
        if record:
            return record["name"]
        return None

# Function to store user's name
def store_user_name(user_id, name):
    with chat_history.driver.session() as session:
        session.run("MERGE (u:User {userID: $user_id}) SET u.name = $name", user_id=user_id, name=name)

# Function to scrape dish names and descriptions based on the selected category
def scrape_dishes(category_filter):
    url = f"https://krua.co/recipe?filter={category_filter}&page=1"
    
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    chromedriver_autoinstaller.install()
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    driver.implicitly_wait(5)
    html = driver.page_source
    mysoup = BeautifulSoup(html, "html.parser")

    dish_divs = mysoup.find_all("div", class_="css-1jdytyu")
    dishes = []
    for dish in dish_divs:
        dish_name_div = dish.find("div", class_="css-f18oi5")
        description_div = dish.find("div", class_="css-g8k6ox")
        
        if dish_name_div and description_div:
            dish_name = dish_name_div.text
            description = description_div.text.strip()
            dishes.append((dish_name, description))
        else:
            print("Dish name or description not found")

    return dishes

# Function to get response from Llama model
def get_llama_response(sentence, user_id):
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    role_prompt = f"""ตอบลูกค้าที่ {get_user_name(user_id)}
    {sentence}
    Answer as briefly as possible as a male chef
    """
    payload = {
        "model": "supachai/llama-3-typhoon-v1.5",
        "prompt": role_prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        response_data = response.text
        data = json.loads(response_data)
        return data.get("response", "ขอโทษด้วย ฉันไม่สามารถให้คำตอบนี้ได้")
    else:
        print(f"Failed to get a response: {response.status_code}, {response.text}")
        return "ขอโทษด้วย ฉันไม่สามารถให้คำตอบนี้ได้"

# Function to compute the best response and handle chat flow
def compute_response(sentence, user_id):
    user_name = get_user_name(user_id)
    previous_answer = chat_history.check_chat_history(user_id, sentence)
    if previous_answer:
        return TextSendMessage(text=previous_answer)
    categories = scrape_categories()

    if "สวัสดี" in sentence:
        if user_name:
            quick_reply_buttons = [
                QuickReplyButton(action=MessageAction(label=cat[:20], text=cat)) for cat in list(categories.keys())[:13]
            ]
            if quick_reply_buttons:
                return TextSendMessage(
                    text=f"สวัสดีครับ, {user_name}! ผมคือ ChefBot ผู้ช่วยเชฟส่วนตัวของคุณ พร้อมแนะนำเมนูอาหารเด็ด ๆ ให้ทุกมื้อ แค่กดที่หมวดหมู่ด้านล่างนี้ที่คุณสนใจ ผมจะช่วยแนะนำเมนูให้คุณทันทีครับ!",
                    quick_reply=QuickReply(items=quick_reply_buttons)
                )
            else:
                return TextSendMessage(text="ขออภัยครับ ไม่พบหมวดหมู่เมนูในขณะนี้ครับ")
        else:
            return TextSendMessage(text="สวัสดีครับ ผมคือ ChefBot ผู้ช่วยเชฟส่วนตัวของคุณ พร้อมแนะนำเมนูอาหารเด็ด ๆ ให้ทุกมื้อ ไม่ว่าจะเป็นเมนูอาหารไทย, ขนมหวาน หรืออาหารนานาชาติ แค่บอกหมวดหมู่ที่ต้องการ ผมจะค้นหาเมนูที่เหมาะกับคุณอย่างรวดเร็วและง่ายดาย มาลองปรุงอาหารให้อร่อยได้ทุกวันกับ ChefBot กันเลยครับ! ก่อนอื่นเลยอยากให้ผมเรียกคุณว่าอะไรดีครับ!?")
        
    if "หิว" in sentence:
        user = get_user_name(user_id)
        quick_reply_buttons = [
            QuickReplyButton(action=MessageAction(label=cat[:20], text=cat)) for cat in list(categories.keys())[:13]
        ]
        if quick_reply_buttons:
            return TextSendMessage(
                text=f"คุณ {user_name} ครับ ทางเรามีหมวดหมู่อาหารหลากหลายให้คุณเลือกตามความต้องการครับ เชิญเลือกได้เลยครับ!",
                quick_reply=QuickReply(items=quick_reply_buttons)
            )
        else:
            return TextSendMessage(text="ขออภัยครับ ไม่พบหมวดหมู่เมนูในขณะนี้ครับ")


    if "ชื่อ" and "อะไร" in sentence:
        user = get_user_name(user_id)
        if user:
            return TextSendMessage(text=f"สวัสดีครับคุณ{user}")
        else :
            return TextSendMessage(text=f"โปรดระบุชื่อของผู้ใช้ เช่น ผมชื่อ{user}")

    if "ชื่อ" in sentence:
        name_match = re.search(r"ชื่อ(.*)", sentence)
        if name_match:
            name = name_match.group(1).strip()
            store_user_name(user_id, name)

            quick_reply_buttons = [
                QuickReplyButton(action=MessageAction(label=cat[:20], text=cat)) for cat in list(categories.keys())[:13]
            ]
            if quick_reply_buttons:
                return TextSendMessage(
                    text=f"ยินดีที่ได้รู้จักครับ, {name}! ผมมีหมวดหมู่อาหารหลากหลายที่น่าสนใจ พร้อมให้คุณเลือกสรรค์ แค่เลือกหมวดหมู่ที่คุณสนใจ แล้วมาค้นพบเมนูอร่อย ๆ ไปด้วยกันครับ!",
                    quick_reply=QuickReply(items=quick_reply_buttons)
                )
            else:
                return TextSendMessage(text="ขออภัยครับ ไม่พบหมวดหมู่เมนูในขณะนี้ครับ")
    
    for cat in categories.keys():
        if cat in sentence:
            dish_info = scrape_dishes(categories[cat])
            dish_info_cache[user_id] = dish_info

            quick_reply_buttons = [
                QuickReplyButton(action=MessageAction(label=dish[0][:20], text=dish[0])) for dish in dish_info[:13]
            ]
            if quick_reply_buttons:
                return TextSendMessage(
                    text=f"นี่คือเมนูอาหารที่น่าสนใจในหมวดหมู่ {cat} ที่คุณเลือกครับ! ลองดูแล้วบอกผมได้เลยว่าเมนูไหนที่ถูกใจ หรือถ้าต้องการคำแนะนำเพิ่มเติม ผมยินดีช่วยเสมอ!",
                    quick_reply=QuickReply(items=quick_reply_buttons)
                )
            else:
                return TextSendMessage(text=f"ไม่พบเมนูในหมวดหมู่ {cat} ที่คุณเลือกครับ")
    
    for dish in dish_info_cache.get(user_id, []):
        if dish[0] in sentence:
            chat_history.store_chat_history(user_id, sentence, f"{dish[0]}: {dish[1]}")
            return TextSendMessage(text=f"{dish[0]}: {dish[1]}")

    # Fallback to Llama response if no category or dish matches
    llama_response = get_llama_response(sentence, user_id)
    chat_history.store_chat_history(user_id, sentence, llama_response)
    return TextSendMessage(text=llama_response)

app = Flask(__name__)

# LINE bot credentials
access_token = 'vc8ZqAx20Fd0+CXx0JaqH8dPP2plIAf6YYig4J1l8sGiDaiyicBIzG4cV+0ELu46OeeybvGp3jIUFhKzcCKnfOnNh2+eUnvggvUxMt8AL5yiC1OfYRCJJ1n90BmXnT+hrxDIV33lOIOQDJiuBU3TTQdB04t89/1O/w1cDnyilFU='
secret = 'e4b3adab91258b1e658cf1cdee43e5a4'
line_bot_api = LineBotApi(access_token)
handler = WebhookHandler(secret)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)

        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        user_id = json_data['events'][0]['source']['userId']

        # Generate response based on the user's message
        response_message = compute_response(msg, user_id)
        line_bot_api.reply_message(tk, response_message)

    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token and secret.")
    except Exception as e:
        print(f"Error: {e}")

    return 'OK'

# Run Flask app
if __name__ == "__main__":
    app.run(port=5002)