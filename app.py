import json
import streamlit as st
import pandas as pd
import joblib
import numpy as np

data_path = "data.json"
model_pipeline = joblib.load('model_pipeline.pkl')

with open(data_path,'r',encoding='utf8') as j:
     data = json.loads(j.read())
     
city_names = [city['name'] for city in data]
city = st.sidebar.selectbox('Şehir', city_names)

selected_city = next(city_data for city_data in data if city_data['name'] == city)
county_names = [county['name'] for county in selected_city['counties']]
county = st.sidebar.selectbox('İlçe', county_names)

neighborhood_names = []
for county_data in selected_city['counties']:
    if county_data['name'] == county:
        for district in county_data['districts']:
            neighborhood_names.extend(neighborhood['name'] for neighborhood in district['neighborhoods'])
neighborhood = st.sidebar.selectbox('Mahalle', neighborhood_names)


apartment_types = ['Bina',
 'Bungalov',
 'Daire',
 'Dağ Evi',
 'Köy Evi',
 'Köşk',
 'Loft Daire',
 'Müstakil Ev',
 'Prefabrik',
 'Residence',
 'Villa',
 'Yalı',
 'Yazlık']

apartment_type = st.sidebar.selectbox('Yapı Tipi', apartment_types)
age=st.sidebar.number_input('Bina Yaşı',0,300,0)
house_size = st.sidebar.number_input('Evin Alanı',50,1000,50)

room_counts = ['3 + 1', '4 + 1', '1 + 1', '2 + 1', '5 + 1', '5 + 2', '7 + 1',
       '6 + 1', '8 + 1', '6 + 2', '10 + 2', '8 + 2', '3 + 2', '7 + 2',
       '4 + 2', '1 + 0', '2 + 0', '2 + 2', '6 + 3', '11 + 2', '9 + 3',
       '6 + 6', '6 + 0', '9 + 1', '25 + 2', '4 + 8', '1 + 3', '3 + 3',
       '3 + 0', '4 + 4', '5 + 3', '1 + 2', '41 + 1', '9 + 2', '10 + 1',
       '11 + 1', '4 + 0', '22 + 2', '4 + 3', '10 + 4', '11 + 3',
       '18 + 18']

room_count = st.sidebar.selectbox('Oda Sayısı', room_counts)
floors= ['Ara Kat', '13. Kat', '4. Kat', 'Çatı Katı', '5. Kat',
       'Bahçe Katı', 'Yüksek Giriş', 'Müstakil', '21 ve üzeri', '3. Kat',
       '1. Kat', '2. Kat', '7. Kat', 'Kot 1', 'Kot 2', 'Giriş Katı',
       'Zemin', '8. Kat', '6. Kat', '9. Kat', 'Teras Katı', 'Kot 3',
       '10. Kat', '18. Kat', '12. Kat', '11. Kat', '14. Kat', '15. Kat',
       'Villa Katı', 'Bodrum', '17. Kat', '19. Kat', '20. Kat', '16. Kat']
floor = st.sidebar.selectbox('Bulunduğu Kat', floors)
furniture = st.sidebar.selectbox("Eşyalı mı?",["Evet","Hayır"])
bath_count = st.sidebar.number_input('Banyo Sayısı',0,50,0)
hand = st.sidebar.selectbox("Kaçıncı Sahibi",["Sıfır","İkinci El"])
heater_types = [
 'Kombi',
 'Klima',
 'Merkezi',
 'Fancoil Ünitesi',
 'Yerden Isıtma',
 'Isıtma Yok',
 'Kat Kaloriferi',
 'Soba',
 'Doğalgaz Sobası',
 'Güneş Enerjisi',
 'Jeotermal Isıtma',
 'Isı Pompası',
 'VRV']

heater_type = st.sidebar.selectbox("Isıtma Tipi",heater_types)
heater_fuels = ['Doğalgaz', 'Elektrik', 'Akaryakıt', 'Kömür']
heater_fuel = st.sidebar.selectbox("Isıtma Yakıtı",heater_fuels)

st.title("Kiralık Ev Fiyatı Tahmini")

st.header("Tahmin Edilmesi İstenen Ev Bilgileri", divider="red")

st.header("")

col1, col2, col3, col4 = st.columns(4)

with col1:
       st.write(f"İl: {city}")
       st.write(f"Bina Yaşı: {age}")
       st.write(f"Eşyalı: {furniture}")
       st.write(f"Yakıt Tipi: {heater_fuel}")
with col2:
       st.write(f"İlçe: {county}")
       st.write(f"Evin Alanı: {house_size}")
       st.write(f"Banyo Sayısı: {bath_count}")
with col3:
       st.write(f"Mahalle: {neighborhood}")
       st.write(f"Oda Sayısı: {room_count}")
       st.write(f"Kaçıncı Sahibi: {hand}")
with col4:
       st.write(f"Yapı Tipi: {apartment_type}")
       st.write(f"Kat: {floor}")
       st.write(f"Isıtma Tipi: {heater_type}")

st.header("",divider="red")

data = {
       "City":[city],
       "Town":[county],
       "Neighborhood":[neighborhood],
       "House Type": [apartment_type],
       "House Age": [age],
       "House Size": [house_size],
       "Room Count": [room_count],
       "Floor": [floor],
       "Furniture":[furniture],
       "Bathroom Count":[bath_count],
       "Hand":[hand],
       "Heater Type": [heater_type],
       "Heater Fuel": [heater_fuel]
}

df = pd.DataFrame(data)

placeholder = st.empty()

if st.button("Tahmin Et"):
       pred = model_pipeline.predict(df)
       if pred:
              st.header(f"Evin Kirası: {pred[0]:.2f}")
       else:
              st.header("Öngörülen değer bulunamadı veya yanlış bir türde.")