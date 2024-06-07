from bs4 import BeautifulSoup
import requests
import joblib
from random import uniform
from time import sleep

user_agent = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/125.0.0.0 Safari/537.36"
}
pages = 1892
start_page = 1
page=1
base_url = "https://www.hepsiemlak.com"
urls = []
while page < pages+1:
    print(f"{page}. sayfa kazınıyor...")

    r = requests.get(f"https://www.hepsiemlak.com/kiralik?page={page}", headers=user_agent)

    if r.status_code == 429:
        print("Request Timeout, sleeping for 120 seconds...")
        sleep(uniform(110,120))
        continue
    
    soup = BeautifulSoup(r.content,"lxml")
    links = [base_url+card['href'] for card in soup.find_all("a",class_ = "card-link")]
    print(f"{len(links)} adet link bulundu.")
    urls.extend(links)
    if urls and page % 10 == 0:
        joblib.dump(urls,f"linkler-{start_page}-{page}.joblib")
        print(f"{len(urls)} url kayıt altına alındı.")
    print(f"{page}. sayfa kazındı.")
    page +=1
    sleep(uniform(10,20))
    