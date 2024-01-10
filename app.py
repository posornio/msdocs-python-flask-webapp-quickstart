import re
from flask import Flask
import pandas
import requests
import numpy as np
from bs4 import BeautifulSoup
from joblib import load


from flask import (Flask, request,jsonify)

app = Flask(__name__)
model = load("GBC_model_account.joblib")


def compter_nombres(chaine):
    pattern = r'\d+'  # Expression régulière pour matcher un ou plusieurs chiffres
    resultats = re.findall(pattern, chaine)
    return len(resultats)


def detecterBots(bio):
    bot = re.findall('bot', bio.lower())
    ai = re.findall(r'\b(AI)\b', bio)
    return len(bot)+len(ai)


def intoToVec(html_content):
    default_img = '/pic/enc/YWJzLnR3aW1nLmNvbS9zdGlja3kvZGVmYXVsdF9wcm9maWxlX2ltYWdlcy9kZWZhdWx0X3Byb2ZpbGVfNDAweDQwMC5wbmc='

    soup = BeautifulSoup(html_content, 'html.parser')
    fullname = soup.select_one('.profile-card-fullname')
    fullname = fullname.text.strip() if fullname else ""

    username = soup.select_one('.profile-card-username')
    username = username.text.strip() if username else ""

    joindate = soup.select_one('.profile-joindate span')
    joindate = joindate['title'] if joindate else 2024
    joindate = joindate[-4:]
    joindate = int(joindate)
    # Extracting additional information
    followers_count = soup.select_one('.followers .profile-stat-num')
    followers_count = int(followers_count.text.replace(',', '')) if followers_count else 0

    following_count = soup.select_one('.following .profile-stat-num')
    following_count = int(following_count.text.replace(',', '')) if following_count else 0

    tweet_count = soup.select_one('.posts .profile-stat-num')
    tweet_count = int(tweet_count.text.replace(',', '')) if tweet_count else 0

    profile_location_span = soup.select_one('.profile-location span:nth-child(2)')
    location = profile_location_span.text.strip() if profile_location_span else ""

    #location = profile_location_span.text.strip() if profile_location_span else ""



# Verification Status
    profile_card_tabs_name = soup.select_one('.profile-card-tabs-name')

    if profile_card_tabs_name:
        verification_icon = profile_card_tabs_name.select_one('.verified-icon')
        is_verified = bool(verification_icon)    # Extracting profile bio and avatar
    profile_bio = soup.select_one('.profile-bio')
    profile_bio = profile_bio.text.strip() if profile_bio else ""

    profile_avatar = soup.select_one('.profile-card-avatar img')
    profile_avatar = profile_avatar['src'] if profile_avatar else ""   
    default_profile_image = default_img==profile_avatar
    ratio_tweet_count = tweet_count/(2024-joindate)
    popularity = np.round(np.log(followers_count+1) * np.log(following_count+1),3)
    word_bot = len(re.findall("bot", profile_bio.lower()))
    hashtag = len(re.findall("#", profile_bio.lower()))
    #descriptionVectorized = process_text_with_word_embeddings(profile_bio)
    #locationVectorized = process_text_with_word_embeddings(location)
    #nameVectorized = process_text_with_word_embeddings(fullname)
    length_description = len(profile_bio)
    contains_pinned_class = bool(soup.select_one('.pinned'))


    nombre_dans_nom = compter_nombres(username)
    word_bot_nom= detecterBots(username)

    data = {'created_at': joindate, 'pinned_tweet_id' : contains_pinned_class, 'default_profile_image':  default_profile_image, 'followers_count': followers_count, 'following_count': following_count, 'tweet_count': tweet_count, 'verfied': is_verified, 'description_length' : length_description ,'name_length' : len(username) ,'ratio_tweet_count' :ratio_tweet_count, 'popularity': popularity, 'word_bot':word_bot, 'hashtag':hashtag,'nombre_dans_nom':nombre_dans_nom, 'word_bot_nom':word_bot_nom}
    print(data)
    return pandas.DataFrame([data])

def predict(y):
    predictProba = model.predict_proba(y)
    predictProba2 = [item[0] for item in predictProba]
    return predictProba2



@app.route('/', methods=['POST'])
def hello():
    name = request.args.get('name')

    ntResponse = requests.get(f"https://nitter.1d4.us/{name}")

    if ntResponse.status_code != 200:
        return jsonify({'error': 'User not found'}), 404
    html_content = ntResponse.text

    y = intoToVec(html_content)

# Open the file in write mode and write the HTML content
   
    #y = preprocessor.transform(y)
    return jsonify({'name': name, 'Bot': predict(y)})



if __name__ == '__main__':
   app.run()



