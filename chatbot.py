from flask import Flask, request, render_template, jsonify, session
import json
import requests
import re
from flask_cors import CORS  
import http.client
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ✅ Allow CORS only from your frontend domain
CORS(app, origins=["https://msmeosem.in"])

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'


def normalize(text):
    """Removes non-alphanumeric characters and converts text to lowercase for consistent matching."""
    return re.sub(r'\W+', '', text.lower())

def fetch_business_data():
    """Fetches business data from the MSME OSEM API."""
    url = "https://msmeosem.in/apis/root/common.php"
    headers = {
        "Accept": "*/*",
        "User-Agent": "Thunder Client",
        "Content-Type": "application/json"
    }
    payload = {"action": "business"}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        json_data = response.json()
        if isinstance(json_data, dict) and 'data' in json_data:
            return json_data['data']
        elif isinstance(json_data, list):
            return json_data
        else:
            print("Business data 'data' key not found or unexpected format.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching business data: {e}")
        return []
    except ValueError as e:
        print(f"Error decoding business data JSON: {e}")
        return []

def fetch_expert_data():
    """Fetches expert data from the MSME OSEM API."""
    conn = http.client.HTTPSConnection("msmeosem.in")
    headers = {
        "Accept": "*/*",
        "User-Agent": "Thunder Client",
        "Content-Type": "application/json"
    }
    payload = json.dumps({"action": "experts"})
    try:
        conn.request("POST", "/apis/root/common.php", payload, headers)
        response = conn.getresponse()
        data = response.read()
        decoded = json.loads(data.decode("utf-8"))
        if isinstance(decoded, list):
            return [item for item in decoded if isinstance(item, dict)]
        elif isinstance(decoded, dict) and 'data' in decoded and isinstance(decoded['data'], list):
            return [item for item in decoded['data'] if isinstance(item, dict)]
        else:
            print("Expert data 'data' key not found or unexpected format.")
            return []
    except Exception as e:
        print(f"Error fetching expert data: {e}")
        return []
    finally:
        conn.close()

def fetch_services():
    """Fetches services data from the MSME OSEM API."""
    conn = http.client.HTTPSConnection("msmeosem.in")
    headers = {
        "Accept": "*/*",
        "User-Agent": "Thunder Client (https://www.thunderclient.com)",
        "Content-Type": "application/json"
    }
    payload = json.dumps({"action": "services"})
    try:
        conn.request("POST", "/apis/root/common.php", payload, headers)
        response = conn.getresponse()
        result = response.read()
        data = json.loads(result.decode("utf-8"))
        return data
    except Exception as e:
        print(f"Error fetching services data: {e}")
        return {"data": []}
    finally:
        conn.close()

def fetch_market_linkage():
    """Fetches market linkage data from the MSME OSEM API using requests."""
    url = "https://msmeosem.in/apis/root/common.php"
    headers = {
        "Accept": "*/*",
        "User-Agent": "Thunder Client",
        "Content-Type": "application/json"
    }
    payload = {"action": "marketLinkage"}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        json_data = response.json()
        if isinstance(json_data, dict) and 'data' in json_data:
            return json_data['data']
        elif isinstance(json_data, list):
            return json_data
        else:
            print("Market linkage data 'data' key not found or unexpected format.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching market linkage data: {e}")
        return []
    except ValueError as e:
        print(f"Error decoding market linkage JSON: {e}")
        return []

def fetch_events():
    """Fetches event data from the MSME OSEM API."""
    conn = http.client.HTTPSConnection("msmeosem.in")
    headers = {
        "Accept": "*/*",
        "User-Agent": "Thunder Client (https://www.thunderclient.com)",
        "Content-Type": "application/json"
    }
    payload = json.dumps({"action": "events"})
    try:
        conn.request("POST", "/apis/root/common.php", payload, headers)
        response = conn.getresponse()
        result = response.read()
        data = json.loads(result.decode("utf-8"))
        return data
    except Exception as e:
        print(f"Error fetching event data: {e}")
        return {"data": []}
    finally:
        conn.close()


try:
    with open("intents.json", "r") as file:
        intents_data = json.load(file)
except FileNotFoundError:
    print("intents.json not found. The chatbot will rely only on API data and hardcoded responses.")
    intents_data = []

valid_cities = ["agra", "kanpur", "lucknow"]

corpus = []
tags = []
responses = {}


def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)


for item in intents_data:
    if all(k in item for k in ("intent", "patterns", "responses")):
        intent = item["intent"]
        processed_patterns = [preprocess_text(p) for p in item["patterns"]]
        corpus.extend(processed_patterns)
        tags.extend([intent] * len(processed_patterns))
        responses[intent] = item["responses"]


vectorizer = TfidfVectorizer()
if corpus:
    X = vectorizer.fit_transform(corpus)
else:
    X = None


def extract_city(text):
    words = preprocess_text(text).split()
    for word in words:
        if word in valid_cities:
            return word
    return None


def is_city_business_query(text):
    text = text.lower()
    keywords = ["how many business", "business in", "number of business"]
    return any(k in text for k in keywords)


def load_chat_history():
    return session.get('chat_history', [])

def save_chat_history(chat_history):
    session['chat_history'] = chat_history

def find_experts_in_district(query_lower, expert_data):
    """
    Extracts the district from the query and finds experts in that district.
    Handles queries like "who is the expert in <district>".
    """
    match = re.search(r'who is the expert in ([a-z\s]+)', query_lower)
    if match:
        district_queried = match.group(1).strip()

        experts_in_district = []
        for expert in expert_data:
            if normalize(expert.get('district', '')) == normalize(district_queried):
                experts_in_district.append(expert.get('name', 'N/A'))

        if experts_in_district:
            return f"The expert(s) in {district_queried.title()} is/are: {', '.join(experts_in_district)}."
        else:
            return f"No expert found in {district_queried.title()}."
    else:
        return "Please specify the district in your query."


def handle_query(query, business_data, expert_data, service_data, market_linkage_data, event_data):
    """
    Processes the user's query and generates a response based on available data.
    Prioritizes API data responses, then intents.json, then general hardcoded responses.
    """
    query_lower = query.lower().strip()


    if query_lower in ['hi', 'hello', 'hlo']:
        return "Hello! 👋 How can I assist you today?"
    elif query_lower in ['how are you', 'how are you doing']:
        return "I'm just a chatbot, but I'm doing great! 😊 How can I help you?"
    elif query_lower in ['who are you', 'what are you']:
        return "I'm your Business Info Chatbot 🤖 here to help you with business-related queries."
    elif query_lower in ['help', 'how can you help me', 'what can you do']:
        return "You can ask me about business or expert info like contacts, counts, locations, and more."

    # Expert
    match = re.search(r"how many expert[s]? in (\w+)", query_lower)
    if match:
        location = match.group(1)
        count = sum(1 for item in expert_data if item.get("district", "").lower() == location.lower())
        return f"There are {count} experts in {location.title()}."

    match = re.search(r"how many expert[s]? in state (\w+)", query_lower)
    if match:
        state = match.group(1)
        count = sum(1 for item in expert_data if item.get("state", "").lower() == state.lower())
        return f"There are {count} experts in state {state.title()}."

    match = re.search(r"designation of ([a-zA-Z\s]+)", query_lower)
    if match:
        name = match.group(1).strip().lower()
        for item in expert_data:
            if normalize(item.get("name", "")) == normalize(name):
                return f"The designation of {name.title()} is {item.get('designation', 'Not available')}."
        return f"Expert named {name.title()} not found."


    for label, key_business, key_expert in [
        ("contact number", "business_contact", "mobile"),
        ("email", "business_email", "email"),
        ("urn", "business_urn", None),
        ("id", "id", "id")
    ]:
        if label in query_lower and 'of' in query_lower:
            name_part = query_lower.split('of', 1)[1].strip().lower()


            for d in business_data:
                if normalize(name_part) in normalize(d.get("business_name", "")):
                    return f"{label.title()} of {d['business_name']}: {d.get(key_business, 'Not available')}"


            if key_expert:
                for e in expert_data:
                    if normalize(name_part) in normalize(e.get("name", "")):
                        return f"{label.title()} of {e['name']}: {e.get(key_expert, 'Not available')}"

            return f"Name '{name_part.title()}' not found in businesses or experts."


    if 'list' in query_lower and 'business' in query_lower:
        if 'lucknow' in query_lower and 'female' in query_lower:
            filtered_businesses = [f"{d['business_name']} - {d['owner_name']}"
                                   for d in business_data if 'lucknow' in d.get('business_district', '').lower() and 'female' in d.get('owner_gender', '').lower()]
            if filtered_businesses:
                return "<br>".join(filtered_businesses)
            return "No female-owned businesses found in Lucknow."
        elif 'manufacturing' in query_lower and 'micro' in query_lower:
            filtered_businesses = [f"{d['business_name']} - {d['business_activity']}"
                                   for d in business_data if 'manufacturing' in d.get('business_activity', '').lower() and 'micro' in d.get('business_industry', '').lower()]
            if filtered_businesses:
                return "<br>".join(filtered_businesses)
            return "No micro manufacturing businesses found."
        elif 'enterprises' in query_lower:
            filtered_businesses = [f"{d['business_name']} - {d['owner_name']}"
                                   for d in business_data if 'enterprises' in d.get('business_name', '').lower()]
            if filtered_businesses:
                return "<br>".join(filtered_businesses)
            return "No enterprises found."
        else:
            if business_data:
                return "<br>".join([f"- {d['business_name']}" for d in business_data])
            return "No businesses found."

    if 'female' in query_lower and 'business' not in query_lower:
        female_owners = [f"{d['business_name']} - {d['owner_name']}"
                         for d in business_data if 'female' in d.get('owner_gender', '').lower()]
        if female_owners:
            return "<br>".join(female_owners)
        return "No female business owners found."
    elif 'male' in query_lower and 'business' not in query_lower:
        male_owners = [f"{d['business_name']} - {d['owner_name']}"
                       for d in business_data if 'male' in d.get('owner_gender', '').lower()]
        if male_owners:
            return "<br>".join(male_owners)
        return "No male business owners found."

    elif 'owner name' in query_lower and 'of' in query_lower:
        business = query_lower.split('of', 1)[1].strip()
        for d in business_data:
            if normalize(business) in normalize(d.get('business_name', '')):
                return f"Owner name of {d['business_name']}: {d.get('owner_name', 'Not available')}"
        return "Business not found."

    elif 'gender' in query_lower and 'of' in query_lower:
        business = query_lower.split('of', 1)[1].strip()
        for d in business_data:
            if normalize(business) in normalize(d.get('business_name', '')):
                return f"Gender of owner: {d.get('owner_gender', 'Not specified')}"
        return "Business not found."

    elif 'how many business in' in query_lower:
        city_match = re.search(r"how many business in (.+)", query_lower)
        if city_match:
            city = city_match.group(1).strip()
            count = sum(1 for d in business_data if city.lower() in d.get('business_district', '').lower())
            return f"There are {count} businesses in {city.title()}."
        return "Please specify the city, e.g., 'how many business in Lucknow'."

    elif "how many business" in query_lower or "total business" in query_lower:
        return f"There are {len(business_data)} businesses in the list."


    if 'list' in query_lower and 'business' in query_lower:
        pass


    match = re.search(r"name the business in ([a-zA-Z\s]+)", query_lower)
    if match:
        district_name = match.group(1).strip()
        found_businesses = [
            f"- {d['business_name']}"
            for d in business_data
            if normalize(d.get('business_district', '')) == normalize(district_name)
        ]
        if found_businesses:
            return f"Here are the businesses in {district_name.title()}:<br>" + "<br>".join(found_businesses)
        else:
            return f"No businesses found in {district_name.title()}."



    elif 'how many business in' in query_lower:
        city_match = re.search(r"how many business in (.+)", query_lower)
        if city_match:
            city = city_match.group(1).strip()
            count = sum(1 for d in business_data if city.lower() in d.get('business_district', '').lower())
            return f"There are {count} businesses in {city.title()}."
        return "Please specify the city, e.g., 'how many business in Lucknow'."

    elif "how many business" in query_lower or "total business" in query_lower:
        return f"There are {len(business_data)} businesses in the list."


    if 'list business' in query_lower:
        if business_data:
            return "<br>".join([f"- {d['business_name']}" for d in business_data])
        return "No businesses found."


    if "who is the owner of" in query_lower:
        match = re.search(r"who is the owner of ([a-zA-Z\s]+)", query_lower)
        if match:
            business_name = match.group(1).strip()
            for business in business_data:
                if normalize(business.get('business_name', '')) == normalize(business_name):
                    return f"The owner of {business['business_name']} is {business.get('owner_name', 'Not Available')}."
            return f"Business '{business_name.title()}' not found."

    if "what is the designation of the owner of" in query_lower:
        match = re.search(r"what is the designation of the owner of ([a-zA-Z\s]+)", query_lower)
        if match:
            business_name = match.group(1).strip()
            for business in business_data:
                if normalize(business.get('business_name', '')) == normalize(business_name):
                    return f"The designation of the owner of {business['business_name']} is {business.get('owner_designation', 'Not Available')}."
            return f"Business '{business_name.title()}' not found."

    if "what is the gender of the owner of" in query_lower:
        match = re.search(r"what is the gender of the owner of ([a-zA-Z\s]+)", query_lower)
        if match:
            business_name = match.group(1).strip()
            for business in business_data:
                if normalize(business.get('business_name', '')) == normalize(business_name):
                    return f"The gender of the owner of {business['business_name']} is {business.get('owner_gender', 'Not Available')}."
            return f"Business '{business_name.title()}' not found."

    if "what is the category of the owner of" in query_lower:
        match = re.search(r"what is the category of the owner of ([a-zA-Z\s]+)", query_lower)
        if match:
            business_name = match.group(1).strip()
            for business in business_data:
                if normalize(business.get('business_name', '')) == normalize(business_name):
                    return f"The category of the owner of {business['business_name']} is {business.get('owner_category', 'Not Available')}."
            return f"Business '{business_name.title()}' not found."

    if "what type of business is" in query_lower:
        match = re.search(r"what type of business is ([a-zA-Z\s]+)", query_lower)
        if match:
            business_name = match.group(1).strip()
            for business in business_data:
                if normalize(business.get('business_name', '')) == normalize(business_name):
                    return f"{business['business_name']} is a {business.get('business_industry', 'Not Available')} business."
            return f"Business '{business_name.title()}' not found."

    if "what is the urn number of" in query_lower:
        match = re.search(r"what is the urn number of ([a-zA-Z\s]+)", query_lower)
        if match:
            business_name = match.group(1).strip()
            for business in business_data:
                if normalize(business.get('business_name', '')) == normalize(business_name):
                    return f"The URN number of {business['business_name']} is {business.get('business_urn', 'Not Available')}."
            return f"Business '{business_name.title()}' not found."

    if "what is the contact number of" in query_lower:
        match = re.search(r"what is the contact number of ([a-zA-Z\s]+)", query_lower)
        if match:
            business_name = match.group(1).strip()
            for business in business_data:
                if normalize(business.get('business_name', '')) == normalize(business_name):
                    return f"The contact number of {business['business_name']} is {business.get('business_contact', 'Not Available')}."
            return f"Business '{business_name.title()}' not found."

    if "what is the email id of" in query_lower:
        match = re.search(r"what is the email id of ([a-zA-Z\s]+)", query_lower)
        if match:
            business_name = match.group(1).strip()
            for business in business_data:
                if normalize(business.get('business_name', '')) == normalize(business_name):
                    return f"The email ID of {business['business_name']} is {business.get('business_email', 'Not Available')}."
            return f"Business '{business_name.title()}' not found."


    if "where is" in query_lower and "located" in query_lower:
        match = re.search(r"where is ([a-zA-Z\s]+) located", query_lower)
        if match:
            business_name = match.group(1).strip()
            for business in business_data:
                if normalize(business.get('business_name', '')) == normalize(business_name):
                    return f"{business['business_name']} is located at {business.get('business_address', 'Not Available')}."
            return f"Business '{business_name.title()}' not found."

    if "in which city is" in query_lower and "situated" in query_lower:
        match = re.search(r"in which city is ([a-zA-Z\s]+) situated", query_lower)
        if match:
            business_name = match.group(1).strip()
            for business in business_data:
                if normalize(business.get('business_name', '')) == normalize(business_name):
                    return f"{business['business_name']} is situated in {business.get('business_district', 'Not Available')}."
            return f"Business '{business_name.title()}' not found."

    if "what is the industry type of" in query_lower:
        match = re.search(r"what is the industry type of ([a-zA-Z\s]+)", query_lower)
        if match:
            business_name = match.group(1).strip()
            for business in business_data:
                if normalize(business.get('business_name', '')) == normalize(business_name):
                    return f"The industry type of {business['business_name']} is {business.get('business_industry', 'Not Available')}."
            return f"Business '{business_name.title()}' not found."

    if "what does" in query_lower and "manufacture" in query_lower:
        match = re.search(r"what does ([a-zA-Z\s]+) manufacture", query_lower)
        if match:
            business_name = match.group(1).strip()
            for business in business_data:
                if normalize(business.get('business_name', '')) == normalize(business_name):
                    return f"{business['business_name']} manufactures {business.get('business_activity', 'Not Available')}."
            return f"Business '{business_name.title()}' not found."

    if "what is the pin code of" in query_lower:
        match = re.search(r"what is the pin code of ([a-zA-Z\s]+)", query_lower)
        if match:
            business_name = match.group(1).strip()
            for business in business_data:
                if normalize(business.get('business_name', '')) == normalize(business_name):
                    return f"The pin code of {business['business_name']} is {business.get('business_pincode', 'Not Available')}."
            return f"Business '{business_name.title()}' not found."


    if "who is the finance expert" in query_lower:
        for expert in expert_data:
            if normalize(expert.get('designation', '')) == normalize("Finance Expert"):
                return f"The finance expert is {expert.get('name', 'N/A')}."
        return "No finance expert found."

    if "what is the email of" in query_lower:
        match = re.search(r"what is the email of ([a-zA-Z\s]+)", query_lower)
        if match:
            expert_name = match.group(1).strip()
            for expert in expert_data:
                if normalize(expert.get('name', '')) == normalize(expert_name):
                    return f"The email of {expert.get('name', 'N/A')} is {expert.get('email', 'N/A')}."
            return f"Expert named '{expert_name.title()}' not found."

    if "what is the contact number of" in query_lower:
        match = re.search(r"what is the contact number of ([a-zA-Z\s]+)", query_lower)
        if match:
            expert_name = match.group(1).strip()
            for expert in expert_data:
                if normalize(expert.get('name', '')) == normalize(expert_name):
                    return f"The contact number of {expert.get('name', 'N/A')} is {expert.get('mobile', 'N/A')}."
            return f"Contact number for expert named '{expert_name.title()}' not found."

    if "what is the designation of" in query_lower:
        match = re.search(r"what is the designation of ([a-zA-Z\s]+)", query_lower)
        if match:
            expert_name = match.group(1).strip()
            for expert in expert_data:
                if normalize(expert.get('name', '')) == normalize(expert_name):
                    return f"{expert.get('name', 'N/A')} is a {expert.get('designation', 'N/A')}."
            return f"Expert named '{expert_name.title()}' not found."


    if "who is the expert in" in query_lower: # This is the specific line to fix
        return find_experts_in_district(query_lower, expert_data)

    
    if "list of expert in" in query_lower: # This is the specific line to fix
        return find_experts_in_district(query_lower, expert_data)

    if "what is the address of the finance expert" in query_lower:
        for expert in expert_data:
            if normalize(expert.get('designation', '')) == normalize("Finance Expert"):
                return f"The address of the finance expert ({expert.get('name', 'N/A')}) is in {expert.get('district', 'N/A')}, {expert.get('state', 'N/A')}."
        return "No address found for the finance expert."


    if re.search(r"(how many|number of)\s+services.*(osem)?", query_lower) or "what services are available" in query_lower or "list services" in query_lower or "what services do you offer" in query_lower:
        try:
            services = service_data.get("data", [])
            if not services:
                return "Sorry, no services found at the moment."

            response_lines = [f"OSEM provides {len(services)} services. Here are the details:"]
            for service in services:
                name = service.get("service") or service.get("service_name") or service.get("name") or "Unnamed Service"
                category = service.get("category", "N/A")
                price = service.get("price", "N/A")
                days = service.get("days", "N/A")
                response_lines.append(f"🔹 {name} (Category: {category}, Price: ₹{price}, Completion Time: {days} days)")

            return "<br>".join(response_lines)

        except Exception as e:
            return f"Error while fetching services: {str(e)}"


    if "list services under the checklist category" in query_lower:
        checklist_services = []
        services = service_data.get("data", [])
        for service in services:
            if normalize(service.get('category', '')) == normalize("Checklist"):
                checklist_services.append(service.get('name', 'N/A'))
        if checklist_services:
            return "Services under the Checklist category are: <br>" + "<br>".join([f"- {s}" for s in checklist_services])
        return "No services found under the Checklist category."


    if "what checklist services do you have" in query_lower:
        checklist_services = []
        services = service_data.get("data", [])
        for service in services:
            if normalize(service.get('category', '')) == normalize("Checklist"):
                checklist_services.append(service.get('name', 'N/A'))
        if checklist_services:
            return "We have the following checklist services: <br>" + "<br>".join([f"- {s}" for s in checklist_services])
        return "No checklist services found."


    if "market linkage" in query_lower:
        ml_data = market_linkage_data
        if not ml_data:
            return "No market linkage programs found at the moment."

        if query_lower == "market linkage" or query_lower == "what market linkages are available":
            response_lines = ["Here are the market linkage programs available:"]
            for item in ml_data:

                name = item.get("ml_name") or item.get("name") or "Unnamed Program"
                description = item.get("ml_desc") or item.get("description") or "No description available."
                response_lines.append(f"🔗 {name}: {description}")
            if len(response_lines) > 1:
                return "<br>".join(response_lines)
            return "No market linkage programs found."


    processed_input = preprocess_text(query)
    if not processed_input:
        return random.choice(responses.get("fallback", ["I'm not sure I understood that. Can you try rephrasing?"]))

    if is_city_business_query(query):
        city = extract_city(query)
        if city:
            count = sum(1 for d in business_data if city.lower() in d.get('business_district', '').lower())
            return f"There are {count} listed businesses in {city.capitalize()}."
        else:
            return "Sorry, this city is not listed in my business data. Please try another city."

    if X is not None:
        user_vec = vectorizer.transform([processed_input])
        sim_scores = cosine_similarity(user_vec, X)
        best_match_index = sim_scores.argmax()
        confidence = sim_scores[0, best_match_index]

        if confidence > 0.3:
            best_intent = tags[best_match_index]
            return random.choice(responses.get(best_intent, ["Sorry, I can't answer that right now."]))

    return random.choice(responses.get("fallback", [
        "I'm not sure I understood that. Can you try rephrasing or asking something else?",
        "My apologies, I don't have information on that. Can I help with something else?",
        "I'm still learning! Could you please ask in a different way?"
    ]))


@app.route("/", methods=["GET"])
def index():
    """Renders the main chatbot HTML page."""
    chat_history = load_chat_history()
    return render_template("index.html", chat_history=chat_history)

@app.route("/chat", methods=["GET", "POST"])
def ask():
    """Handles chat requests, processes user queries, and returns responses."""
    if request.method == "GET":
        query = request.args.get('message', '')
    else:
        if request.is_json:
            query = request.json.get('message', '')
        else:
            query = request.form.get('query', '')

    query = query.strip()
    if not query:
        return jsonify({'response': "Please enter a message."})

    business_data = fetch_business_data()
    expert_data = fetch_expert_data()
    service_data = fetch_services()
    market_linkage_data = fetch_market_linkage()
    event_data = fetch_events()

    response = handle_query(query, business_data, expert_data, service_data, market_linkage_data, event_data)

    chat_history = load_chat_history()
    chat_history.append({'role': 'user', 'text': query})
    chat_history.append({'role': 'bot', 'text': response})
    save_chat_history(chat_history)

    return jsonify({'response': response})

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    """Clears the chat history from the session."""
    session.pop('chat_history', None)
    return jsonify({"response": "Chat history cleared!"})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
