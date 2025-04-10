import requests, time, json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

BASE_URL  = "https://dustinabbott.net"
START_URL = f"{BASE_URL}/category/photography-reviews/"
HEADERS   = {"User-Agent": "Mozilla/5.0 (compatible; MyBot/1.0)"}

def fetch_soup(url):
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def parse_review_list_page(soup):
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if (
            href.startswith(BASE_URL)
            and "/20" in urlparse(href).path
            and href.endswith("-review/")
        ):
            links.add(href)
    next_btn = soup.find("a", string=lambda t: t and "Older Reviews" in t)
    next_url = urljoin(BASE_URL, next_btn["href"]) if next_btn else None
    return list(links), next_url

def parse_review_page(soup):
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else "No Title"
    time_tag = soup.find("time", datetime=True)
    date = time_tag["datetime"] if time_tag else None
    content_div = soup.find("div", class_="entry-content") or soup.find("article") or soup
    paras = content_div.find_all("p")
    content_text = "\n\n".join(p.get_text(strip=True) for p in paras) if paras else content_div.get_text("\n\n", True)
    return {"title": title, "date": date, "content_text": content_text}

def scrape_all_reviews():
    seen = set()
    all_reviews = []
    next_page = START_URL

    while next_page:
        print(f"Fetching list page: {next_page}")
        list_soup = fetch_soup(next_page)
        review_links, next_page = parse_review_list_page(list_soup)

        for link in review_links:
            if link in seen:
                continue
            seen.add(link)

            print(f"  → Fetching review: {link}")
            rev_soup = fetch_soup(link)
            rec = parse_review_page(rev_soup)
            rec["url"] = link
            all_reviews.append(rec)

            time.sleep(1.0)  # politeness

        time.sleep(2.0)

    return all_reviews

if __name__ == "__main__":
    reviews = scrape_all_reviews()
    with open("dustin_photography_reviews.json", "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    print(f"Scraped {len(reviews)} unique text‐only reviews.")
