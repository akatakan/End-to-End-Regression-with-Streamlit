from pprint import pprint
import joblib as jb
import requests
from bs4 import BeautifulSoup
from time import sleep
from random import uniform
import pandas as pd
import numpy as np

with open("proxy-list.txt","r") as f:
    text = f.readlines()
    text = [str(link).strip("\n") for link in text]
    
proxies = {
    'http': text
}

user_agent = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/125.0.0.0 Safari/537.36"
}

house_links = jb.load("linkler-1-420.joblib")
prices= []
house_types = []
house_sizes = []
house_ages = []
apartment_types = []
heater_types = []
floors = []
with_furnitures = []
bath_counts = []
hand_nums = []
heater_fuels = []
cities = []
towns = []
neighborhoods = []
room_types = []


def get_info(soup):
    
    
    ul = soup.find("ul",class_="short-info-list")
    if ul:
        short_list = ul.find_all('li')
    price = soup.find("p",class_ = "fz24-text price").text.strip().split()[0]
    info_list = soup.find_all("span",class_="txt")
    
    city = short_list[0].text.strip()
    town = short_list[1].text.strip()
    neighborhood = short_list[2].text.strip()
    house_type = short_list[4].text.strip()
    room_type = short_list[5].text.strip()
    house_size = short_list[6].text.strip() 
    price = int(price.replace('.',''))
    
    floor = None
    house_age = None
    heater_type = None
    with_furniture = None
    bath_count = None
    hand_num = None
    heater_fuel = None
    apartment_type = None
    
    
    for i in range(len(info_list)):
        if info_list[i].text == "Bulunduğu Kat":
            floor = info_list[i].find_next_sibling('span').text
        elif info_list[i].text == "Bina Yaşı":
            house_age = info_list[i].find_next_sibling('span').text.split(' ')[0]
        elif info_list[i].text == "Isınma Tipi":
            heater_type = info_list[i].find_next_sibling('span').text
        elif info_list[i].text == "Eşya Durumu":
            with_furniture = True if info_list[i].find_next_sibling("span").text == "Eşyalı" else False
        elif info_list[i].text == "Banyo Sayısı":
            bath_count = info_list[i].find_next_sibling("span").text
        elif info_list[i].text == "Yapının Durumu":
           hand_num = info_list[i].find_next_sibling("span").text
        elif info_list[i].text == "Yakıt Tipi":
            heater_fuel = info_list[i].find_next_sibling("span").text
        elif info_list[i].text == "Konut Tipi":
            apartment_type = info_list[i].find_next_sibling("span").text
    
    prices.append(price)
    house_types.append(house_type)
    room_types.append(room_type)
    house_sizes.append(house_size)
    floors.append(floor)
    house_ages.append(house_age)
    heater_types.append(heater_type)
    with_furnitures.append(with_furniture)
    bath_counts.append(bath_count)
    hand_nums.append(hand_num)
    heater_fuels.append(heater_fuel)
    apartment_types.append(apartment_type)
    cities.append(city)
    towns.append(town)
    neighborhoods.append(neighborhood)
    
step = 0
while step < len(house_links)-1:
    try:
        print(f"{step+1}.link")
        r = requests.get(house_links[step],headers=user_agent,proxies=proxies)
        print(r.status_code)
        if r.status_code == 429:
            print("Timeout...")
            sleep(uniform(120,140))
            continue
        
        soup = BeautifulSoup(r.content,'lxml')
        if soup.find('p',class_="stale-warning__text"):
            step +=1
            continue
        ul = soup.find("ul",class_="short-info-list")
        if not ul:
            step+=1
            continue
        get_info(soup)
        print(f"{step+1}. kazındı.")
        step+=1
        sleep(uniform(1,5))
    except Exception as e:
        print(e)
        print("Bir hata oluştu...")
    
data = {
    "City": cities,
    "Town": towns,
    "Neighborhood": neighborhoods,
    "Apartment Type": apartment_types,
    "House Type": house_types,
    "House Age": house_ages,
    "House Size": house_sizes,
    "Room Count": room_types,
    "Floor": floors,
    "Furniture": with_furnitures,
    "Bathroom Count": bath_counts,
    "Hand": hand_nums,
    "Heater Type":heater_types,
    "Heater Fuel":heater_fuels,
    "Price":prices
}


df = pd.DataFrame(data)

df.to_csv("emlak.csv",index=False)