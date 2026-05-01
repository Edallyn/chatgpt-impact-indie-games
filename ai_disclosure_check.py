import requests, time, json
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

COOKIES = {
    "birthtime": "0",
    "lastagecheckage": "1-0-1990",
    "mature_content": "1",
}

def get_ai_disclosure_html(app_id):
    url = f"https://store.steampowered.com/app/{app_id}/"
    try:
        r = requests.get(url, headers=HEADERS, cookies=COOKIES, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup.find_all(["h2", "h3", "b", "strong"]):
            if "AI Generated Content Disclosure" in tag.get_text():
                paragraphs = []
                for sib in tag.find_next_siblings():
                    if sib.name in ["h2", "h3"]:
                        break
                    if sib.name in ["p", "li"]:
                        t = sib.get_text(separator=" ", strip=True)
                        if t:
                            paragraphs.append(t)
                return " ".join(paragraphs) if paragraphs else "disclosure_found"
        return None
    except:
        return None

def get_generative_ai_info(app_id):
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc=us&l=en"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()[str(app_id)]
        if not data.get("success"):
            return None
        return data["data"].get("generative_ai_info", None)
    except:
        return None

try:
    with open("ai_disclosure_results.json", encoding="utf-8") as f:
        results = json.load(f)
    print(f"Checkpoint: {len(results)} oyun zaten tamamlandı")
except:
    results = {}
    print("Sıfırdan başlıyor...")

with open("steam_indie_combined.json", encoding="utf-8") as f:
    df = pd.DataFrame(json.load(f))

done_ids = set(results.keys())
todo = df[~df["appid"].astype(str).isin(done_ids)]
print(f"Kalan: {len(todo)} oyun (~{len(todo)*1.5/3600:.1f} saat)\n")

for i, (_, row) in enumerate(todo.iterrows()):
    appid = int(row["appid"])

    ai_info = get_generative_ai_info(appid)

    html_disclosure = None
    if ai_info is None:
        html_disclosure = get_ai_disclosure_html(appid)

    has_ai = ai_info is not None or html_disclosure is not None

    results[str(appid)] = {
        "name"              : row["name"],
        "post_chatgpt"      : row["post_chatgpt"],
        "generative_ai_info": ai_info,
        "html_disclosure"   : html_disclosure,
        "has_ai_flag"       : int(has_ai),
    }

    if (i + 1) % 50 == 0:
        with open("ai_disclosure_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        ai_count = sum(v["has_ai_flag"] for v in results.values())
        print(f"  {len(results)}/{len(df)} — AI flagli: {ai_count}")

    time.sleep(1.5)

with open("ai_disclosure_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

flagged = sum(v["has_ai_flag"] for v in results.values())
print(f"\nToplam taranan : {len(results)}")
print(f"AI flagli      : {flagged} ({flagged/len(results):.1%})")