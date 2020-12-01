#import packages
import pandas as pd  
import numpy as np
from geopy.exc import GeocoderTimedOut 
from geopy.geocoders import Nominatim
import streamlit as st
from pulp import *
import matplotlib.pyplot as plt
import seaborn as sn
import sys
from streamlit import cli as stcli
from PIL import Image
from IPython.display import display
import math
import pydeck as pdk
import hashlib
import sqlite3
import hashlib
#################################################
st.title('TRONSPORT MANAGMENT SYSTEM')
st.write(""" 
# Welcome to our website  
choose the best way to deliver your services
""")
st.sidebar.write("""
# Parameters
""")

image = Image.open(r'C:\Users\amin\Downloads\tms.jpg')
st.image(image, caption='',use_column_width=True)

##################################
conn = sqlite3.connect('data.db')
c = conn.cursor()
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data
###############################################
option = st.sidebar.selectbox('How would you like to be contacted?',('User', 'Boss'))
st.sidebar.write('You selected:', option)

if option=='User':
    mode = st.sidebar.selectbox('Log in or Sign up ',('Log in', 'Sign up'))
    if mode =='Log in':
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Log in"):
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:
                st.sidebar.success("Logged In as {}".format(username))
                DDD = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],columns=['lat', 'lon'])
                st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',initial_view_state=pdk.ViewState(latitude=37.76,longitude=-122.4,zoom=11,pitch=50,),layers=[pdk.Layer('HexagonLayer',data=DDD,get_position='[lon, lat]',radius=200,elevation_scale=4,elevation_range=[0, 1000],pickable=True,extruded=True,),pdk.Layer('ScatterplotLayer',data=DDD,get_position='[lon, lat]',get_color='[200, 30, 0, 160]',get_radius=200,),],))
                try:
                    adresses = st.sidebar.text_area("Put the number of sales , the center, the other cities splited by ")
                    word=adresses.split(",")
                    K=int(word[0])
                    adre=word[1:]    
                    mdina=adre[0]
                    data = {'City':[]}
                    df = pd.DataFrame(data) 
                    n=len(adre)
                    for i in range(n):
                        df = df.append({'City':adre[i]}, ignore_index=True)
                    longitude = [] 
                    latitude = []
                    def findGeocode(city):
                        try:  
                            geolocator = Nominatim(user_agent="your_app_name")   
                            return geolocator.geocode(city) 
                        except GeocoderTimedOut:   
                            return findGeocode(city)
                    for i in (df["City"]): 
                        if findGeocode(i) != None:     
                            loc = findGeocode(i)  
                            latitude.append(loc.latitude) 
                            longitude.append(loc.longitude)   
                        else: 
                            latitude.append(np.nan) 
                            longitude.append(np.nan)
                    df["Longitude"] = longitude 
                    df["Latitude"] = latitude
                    df.set_index('City', inplace=True) 
                    sites = df.index
                    positions = dict( ( city, (df.loc[city, 'Longitude'], df.loc[city, 'Latitude']) ) for city in sites)
                    def fun_distance(lat1, lon1, lat2, lon2):
                        radius = 6371 # km
                        dlat = math.radians(lat2-lat1)
                        dlon = math.radians(lon2-lon1)
                        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
                              * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
                        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                        d = radius * c

                        return d
                    A=np.zeros((len(df['Latitude']),len(df['Latitude'])))
                    for i in range(len(df['Latitude'])-1):
                        for j in range(i+1,len(df['Latitude'])):
                            A[i][j]=fun_distance(df['Latitude'].iloc[i],df['Longitude'].iloc[i],df['Latitude'].iloc[j],df['Longitude'].iloc[j])
                    DISTANCE =np.zeros((len(df['Latitude']),len(df['Latitude'])))
                    for i in range(len(df['Latitude'])):
                        for j in range(len(df['Latitude'])):
                            if j>i:
                                DISTANCE[i][j]=A[i][j]
                            if j<i:    
                                DISTANCE[i][j]=A[j][i]
                    a=np.matrix(DISTANCE)
                    Distance = pd.DataFrame(a)
                    Distance.index=adre
                    Distance.columns = adre
                    st.write('Distances between cities')
                    st.table(Distance)
                    distances = dict( ((s1,s2), Distance.loc[s1, s2] ) for s1 in positions for s2 in positions if s1!=s2)
                    prob=LpProblem("vehicle", LpMinimize)
                    x = LpVariable.dicts('x',distances, 0,1,LpBinary)
                    u = LpVariable.dicts('u', sites, 0, len(sites)-1, LpInteger)
                    cost = lpSum([x[(i,j)]*distances[(i,j)] for (i,j) in distances])
                    prob+=cost
                    for k in sites:
                        cap = 1 if k != mdina else K
                        prob+= lpSum([ x[(i,k)] for i in sites if (i,k) in x]) ==cap
                        prob+=lpSum([ x[(k,i)] for i in sites if (k,i) in x]) ==cap
                    N=len(sites)/K
                    for i in sites:    
                        for j in sites:
                            if i != j and (i != mdina and j!= mdina) and (i,j) in x:
                                prob += u[i] - u[j] <= (N)*(1-x[(i,j)]) - 1
                    prob.solve()
                    st.write(LpStatus[prob.status])                    
                    non_zero_edges = [ e for e in x if value(x[e]) != 0 ]
                    def get_next_site(parent):
                        edges = [e for e in non_zero_edges if e[0]==parent]
                        for e in edges:
                             non_zero_edges.remove(e)
                        return edges       
                    tours = get_next_site(mdina)
                    tours = [ [e] for e in tours ]
                    for t in tours:
                        while t[-1][1] !=mdina:
                            t.append(get_next_site(t[-1][1])[-1])
                    st.write("The roads to be followed are : ")
                    for t in tours:
                        st.write(' -> '.join([ a for a,b in t]+[mdina]))
                    st.write('Total distance:', value(prob.objective), '(km)')    
                except:
                     st.write('enter ')
            else:
                st.sidebar.warning("Incorrect Username/Password")
    elif mode == "Sign up":
        st.subheader("Create New Account")
        new_user = st.sidebar.text_input("Username")
        new_password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.button("Sign up"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
   
   
   