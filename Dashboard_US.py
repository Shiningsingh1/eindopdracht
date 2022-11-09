#import packages
#pip install geopandas
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import folium
import matplotlib.pyplot as plt
import plotly.express as px     
import numpy as np
import requests
import folium
import leafmap.foliumap as leafmap
import base64
import hydralit_components as hc
import webbrowser
import datetime as dt
import statsmodels.api as ols
import plotly.graph_objects as go
from PIL import Image
from streamlit_option_menu import option_menu
from geopy.geocoders import Nominatim
from sklearn import linear_model
from statsmodels.formula.api import ols
import mpl_toolkits
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')



#achtergrond foto
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
    background-image:url("http://wallup.net/wp-content/uploads/2017/11/17/228112-blurred-mountain-sun_rays.jpg");
    background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


#data inladen, kolommen hernoemen, droppen
df = pd.read_csv("shootings_data.csv")
df = df.drop(["Unnamed: 0"], axis=1)
df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')
df["death"]= 1
df = df.rename({'name': 'Name', 'date': 'Date',
                'manner_of_death':'Manner of death',
                "armed":"Armed?", "age":"Age", "gender":"Gender",
                "race":"Race", "city":"City", "state":"State",
                "signs_of_mental_illness":"Signs of mental illness",
               "threat_level":"Threat level", "flee":"Flee",
               "arms_category":"Arms category", "year":"Year"}, axis=1) 
deathsperyear = pd.read_csv("Deaths per year.csv")


#slider
if st.sidebar.checkbox("Sliders"):
 _age = st.sidebar.slider('Selecteer een leeftijd', df['Age'].min(), df['Age'].max(), value=(18.0000))
 df = df[df['Age'] <= _age]


#intropagina
def intro():
    st.title("US Police killings from 2015 - 2020")
    url = "https://www.kaggle.com/datasets/ahsen1330/us-police-shootings"
    linkedin_jaskirat="linkedin.com/in/jaskiratdhillononly/"
    linkedin_thijs="https://www.linkedin.com/in/thijsvandenberg2003/"
    image = Image.open('jaski.jpg')
    image2 = Image.open('thijs.jpg')


    col3, col4, col5 = st.columns(3)
    with col3:
     if st.button('Kaggle dataset'):
        webbrowser.open_new_tab(url)

    with col4:
     if st.button('LinkedIn Jaskirat'):
        webbrowser.open_new_tab(linkedin_jaskirat)

    with col5:
     if st.button('LinkedIn Thijs'):
        webbrowser.open_new_tab(linkedin_thijs)
    

    col1, col2 = st.columns(2)
    with col1:
     st.image(image, width=200)
     st.subheader("Jaskirat Singh Dhillon")
    with col2: 
     st.image(image2, width=200)
     st.subheader("Thijs van den Berg")


def dataset():
   df = pd.read_csv("shootings_data.csv")
   df = df.drop(["Unnamed: 0"], axis=1)
   df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')
   df["death"]= 1
   df = df.rename({'name': 'Name', 'date': 'Date',
                'manner_of_death':'Manner of death',
                "armed":"Armed?", "age":"Age", "gender":"Gender",
                "race":"Race", "city":"City", "state":"State",
                "signs_of_mental_illness":"Signs of mental illness",
               "threat_level":"Threat level", "flee":"Flee",
               "arms_category":"Arms category", "year":"Year"}, axis=1) 
   st.write(df)


def linechart1():
   data = df.groupby(['year_month'])['Name'].count().reset_index()
   data.columns = ['year_month', 'count']
   data['year_month'] = data['year_month'].astype(str)
   data = data.head(65)

   fig = px.line(
   data, 
   x="year_month", 
   y="count", 
   title='Doden per jaar per maand',
   labels={"year_month":"Jaar", "count":"Aantal doden"})

   st.write(fig)


def histogram1():
   state_selection = df["State"].unique().tolist()
   states = st.selectbox("Selecteer een staat", state_selection)
   race_color_map = {"Asian": "maroon", "White": "gold","Hispanic" : "deepskyblue" ,"Black": "springgreen","Other": "crimson", "Native" : "lightpink"}
   s = df[df["State"] == states]

   col1, col2 = st.columns(2)
   with col1:
    fig = px.histogram(s,
                   x="Race", 
                   labels={"Race":"Afkomst", "count":"Aantal doden"},
                   title= "Verdeling afkomst van slachtoffers",                       
                   color="Race",
                   hover_name='Race',
                   color_discrete_map = race_color_map)
    fig.update_layout()  
    st.write(fig)


   with col2:
    fig = px.histogram(data_frame= s, x="Age",
                  labels={"Age": "Leeftijd in jaren", "count":"Aantal doden", "Race":"Afkomst"},
                  title= "Verdeling leeftijd per ras",
                  color="Race",
                  color_discrete_map = race_color_map)
    fig.update_layout()

    st.write(fig)



def barchart1():
   col1, col2 = st.columns(2)

   with col1:
    city = df.groupby('City')['id'].count().reset_index().sort_values('id', ascending=True).tail(10)
    fig = px.bar(city, x="id", y="City", title="Top 10 steden met meeste doden",
           labels={"id": "Doden", "city":"Stad"})

    st.write(fig)

   with col2:
    state = df.groupby('State')['id'].count().reset_index().sort_values('id', ascending=True).tail(5)
    fig = px.bar(state, x="id", y="State", title="Top 5 staten met meeste doden",
           labels={"id": "Doden","State":"Staat"})
   
    st.write(fig)



def boxplot():    
   col1, col2 = st.columns(2)
   with col1:      
    fig = px.box(df, 
             x="Gender", 
             y="Age",
             labels={"M": "Male",
                     "F": "Female",
                     "Gender":"Geslacht",
                     "Age":"Leeftijd"},                     
             title="Verdeling slachtoffers over leeftijd",                                  
             color="Gender")                              
    st.plotly_chart(fig)

   with col2:      
    fig = px.box(df, 
             x="Race", 
             y="Age",                     
             title="Verdeling slachtoffers over ras",                                  
             color="Race",
             labels={"Race": "Afkomst",
                     "Age":"Leeftijd"})                              
    st.plotly_chart(fig)


def map1(): 
     mapdata = pd.DataFrame(df.groupby(["State",  "Race", "lat state", "lon state", "Age", "Year"])["Manner of death"].count())
     mapdata = mapdata.reset_index()
   
     race_selection = mapdata["Race"].unique().tolist()
     races = st.selectbox("Selecteer een ras", race_selection)
     mapdata = mapdata[mapdata["Race"] == races]

     m = leafmap.Map(center=[50, -110], zoom=3)
     m.add_basemap("SATELLITE")
     polygons = 'https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_states.json'
     m.add_geojson(polygons, layer_name="Countries")

     m.add_heatmap(mapdata,
             latitude="lat state",
             longitude="lon state",
             name="Heat map",
             value="Manner of death",
             radius=25)

     m.to_streamlit(width=700, height=500, add_layer_control=True)

def map2():
     df1 = pd.DataFrame(df.groupby(["City",  "Race", "lat city", "lon city"])["Manner of death"].count())
     df1 = df.reset_index()
   
     race_selection = df1["Race"].unique().tolist()
     races = st.selectbox("Selecteer een ras", race_selection)
     mapdata2 = df1[df1["Race"] == races]

     m1 = leafmap.Map(location=[36.778259, -119.417931], zoom=5.7)
     m1.add_basemap("SATELLITE")
     polygons = 'https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_states.json'
     m1.add_geojson(polygons, layer_name="Countries")
 
     m1.add_points_from_xy(mapdata2,
                     x="lon city",
                     y="lat city")
 
     m1.to_streamlit(width=700, height=500, add_layer_control=True)

def map3():
   danger = df.loc[df["Race"] == "Other"]
   danger["Terrorist?"] = ["No" if x == False else "Yes" for x in danger["Signs of mental illness"]]
   danger.dropna()
   st.write(danger)

   col1, col2 = st.columns(2)
   with col1:
    selection = danger["Terrorist?"].unique().tolist()
    value = st.selectbox("Maak uw selectie", selection)
    s = danger[danger["Terrorist?"] == value]
   
   with col2: 
    m2 = leafmap.Map(center=[50, -110], zoom=3)
    m2.add_basemap("SATELLITE")
    polygons = 'https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_states.json'
    m2.add_geojson(polygons, layer_name="Countries")
    m2.add_points_from_xy(s,
                     x="lon city",
                     y="lat city")           
    m2.to_streamlit(width=300, height=500, add_layer_control=True)

   

   fig = px.bar(s, x="Name", y="Age", title="Naam en leeftijd verdachten", labels={'Name': 'Naam', 'Age':'Leeftijd'})
   st.write(fig)


def model():
   data1 = df.groupby(["Date"])['Name'].count().reset_index()
   data1.columns = ['datum', 'count']
   data1['datum'] = pd.to_datetime(data1['datum'])
   data1['datum']=data1['datum'].map(dt.datetime.toordinal)
   doden_datum = ols("count ~ datum", data=data1).fit()
   explanatory= pd.DataFrame({'datum': np.arange(737591,738000)})
   test = doden_datum.predict(explanatory)
   prediction = explanatory.assign(test = doden_datum.predict(explanatory))


   fig=plt.figure(figsize= (25,12))
   sns.regplot(x='datum', y='count', ci=None, data=data1)
   sns.scatterplot(x='datum', y='test', data=prediction, color='blue', marker="s")
   plt.xlabel('Datum in numerieke format')
   plt.ylabel('Aantal doden')
   plt.title('Aantal doden per numerieke datum')

   st.pyplot(fig)


#Navigationbar op pagina's
def m():
 menu_data = option_menu(menu_title="Selecteer een map",
                         options=["Map: US Police Killings", 
                                    "Map: US Police Killings in California",],
                         orientation="horizontal")

 if menu_data == "Map: US Police Killings":
  map1()
 elif menu_data == "Map: US Police Killings in California":
  map2()


def m2():
 menu_data = option_menu(menu_title="Selecteer een visualisatie",
                         options=["Histogram: inzicht per ras en leeftijd", 
                                  "Grafiek: aantal doden per jaar per maand"],
                         orientation="horizontal")

 if menu_data == "Histogram: inzicht per ras en leeftijd":
  histogram1()
 elif menu_data == "Grafiek: aantal doden per jaar per maand":
  linechart1()



#Sidebar
with st.sidebar:
    selected = option_menu(menu_title="Maak een selectie",
                           options=["Welkom!",
                                    "Dataset: US Police Killings",
                                    "Top 10 steden/staten",
                                    "Inzicht slachtoffers per ras", 
                                    "Verdeling slachtoffers over leeftijd en ras",
                                    "Map: aantal doden in VS",
                                    "Mogelijke gevaar van aanslagen",
                                    "Prediction"],
                                    )

#Koppelen visualisaties met sidebar    
if selected == "Welkom!":
       intro()
elif selected == "Dataset: US Police Killings":
   dataset()
elif selected == "Top 10 steden/staten":
      barchart1() 
elif selected == "Inzicht slachtoffers per ras":
       m2()
elif selected =="Verdeling slachtoffers over leeftijd en ras":
       boxplot()
elif selected =="Map: aantal doden in VS":
       m()
elif selected == "Mogelijke gevaar van aanslagen":
      map3()
elif selected == "Prediction":
      model()

