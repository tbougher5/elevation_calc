import streamlit as st
import numpy as np
import pandas as pd
#import plotly.graph_objects as go
import pickle
#import math
import matplotlib.pyplot as plt
#import os
#import pdfkit
from streamlit_drawable_canvas import st_canvas
#from PIL import Image
import json

# st.markdown("""
# <style>
# @font-face{
#     font-family: 'Roboto Condensed';
#     font-style: normal;
#     font-weight: 400
#     src: url(RobotoCondensed-Regular.ttf) format('truetype')
# }
# html,body, [class*="css"]{
#     font-family: 'Roboto Condensed';
#     font-size: 24px;
# }
# </style>
# """,
# unsafe_allow_html=True,
# )

def make_json(w,h,x,y,s,app):
    if app == False:
        json_data = {
        "version": "4.4.0",
        "objects": [
            {
            "type": "rect",
            "originX": "left",
            "originY": "top",
            "left": x,
            "top": y,
            "width": w,
            "height": h, 
            "fill": "rgba(225, 225, 255, 1.0)",
            "stroke": "#8C8C8C",
            "strokeWidth": s,
            "strokeDashArray": None,
            "strokeLineCap": "butt",
            "strokeDashOffset": 0,
            "strokeLineJoin": "miter",
            "strokeUniform": True,
            "strokeMiterLimit": 4,
            "scaleX": 1,
            "scaleY": 1,
            "angle": 0,
            "flipX": False,
            "flipY": False,
            "opacity": 1,
            "shadow": None,
            "visible": True,
            "backgroundColor": "",
            "fillRule": "nonzero",
            "paintFirst": "fill",
            "globalCompositeOperation": "source-over"
            }
        ]
        }
    else:
        json_data = {
            "type": "rect",
            "originX": "left",
            "originY": "top",
            "left": x,
            "top": y,
            "width": w,
            "height": h, 
            "fill": "rgba(225, 225, 255, 1.0)",
            "stroke": "#8C8C8C",
            "strokeWidth": s,
            "strokeDashArray": None,
            "strokeLineCap": "butt",
            "strokeDashOffset": 0,
            "strokeLineJoin": "miter",
            "strokeUniform": True,
            "strokeMiterLimit": 4,
            "scaleX": 1,
            "scaleY": 1,
            "angle": 0,
            "flipX": False,
            "flipY": False,
            "opacity": 1,
            "shadow": None,
            "visible": True,
            "backgroundColor": "",
            "fillRule": "nonzero",
            "paintFirst": "fill",
            "globalCompositeOperation": "source-over"
        }
    return(json_data)

#fileDir = 'C:\Python\Deploy'
pFile = 'Product_DF_v2.sav'
dfP = pd.read_pickle(pFile)

st.set_page_config(layout="wide")

#def run_calc(ht, wd, rep):
st.text("")
st.title('OBE EMBODIED CARBON CALCULATOR')
st.header('FENESTRATION ELEVATION DIAGRAM')
st.text("")

st.markdown(f'''
    <style>
        section[data-testid="stSidebar"] .css-ng1t4o {{width: 28rem;}}
        section[data-testid="stSidebar"] .css-1d391kg {{width: 28rem;}}
    </style>
''',unsafe_allow_html=True)

with st.sidebar.header('Openings'):
    st.checkbox(label='Compare 2 Products', key='win2', value=True)
    col1, col2  = st.sidebar.columns((1,1))
    col1.subheader('Product #1 Details')
    col2.subheader('Product #2 Details')

    col1.selectbox('Product Type', ('Curtainwall', 'Window wall', 'Store front'),\
        index=0, key='type1', help=None, on_change=None, args=None, kwargs=None, disabled=False)
    if st.session_state.type1:
        frames = dfP.loc[(dfP['Product type']==st.session_state.type1)]
        if st.session_state.type1 == 'Generic U-value':
            uval1_lbl = 'Total product U-value'
        else:
            uval1_lbl = 'COG U-value' 
    if st.session_state.win2:
        col2.selectbox('Product Type', ('Curtainwall', 'Window wall', 'Store front'),\
            index=0, key='type2', help=None, on_change=None, args=None, kwargs=None, disabled=False)
        if st.session_state.type2:
            frames2 = dfP.loc[(dfP['Product type']==st.session_state.type2)]

            if st.session_state.type2 == 'Generic U-value':
                uval2_lbl = 'Total product U-value'
            else:
                uval2_lbl = 'COG U-value' 
        
    col1.selectbox('OBE Product', frames,\
        index=0, key='prod1', help=None, on_change=None, args=None, kwargs=None, disabled=False)
        
    if st.session_state.win2:
        col2.selectbox('OBE Product', frames2,\
        index=0, key='prod2', help=None, on_change=None, args=None, kwargs=None, disabled=False)

    col1.radio('Glass Type (for embodied carbon only)',('Double pane','Triple pane'), key='glass1')
    if st.session_state.win2:
        col2.radio('Glass Type (for embodied carbon only)',('Double pane','Triple pane'), key='glass2')

    col1.checkbox(label='Manually enter embodied carbon', key='man_ec1')
    if st.session_state.man_ec1:
        col1.number_input(label='Total GWP, frame + glass [kgCO2/m²]', value = 150, min_value = 0, max_value = 1000, step = 10, format = "%i", key = 'ec1')

    if st.session_state.win2:
        col2.checkbox(label='Manually enter embodied carbon', key='man_ec2')
        if st.session_state.man_ec2:
            col2.number_input(label='Total GWP, frame + glass [kgCO2/m²]', value = 150, min_value = 0, max_value = 1000, step = 10, format = "%i", key = 'ec2') 
    col1, col2  = st.sidebar.columns((1,1))

    n_h = col1.slider(label='Horizontal Openings', min_value=1,max_value=5, value=5, step = 1)
    n_v = col2.slider(label='Vertical Openings', min_value=1,max_value=5, value=2)

    stroke_width = 5 #= st.sidebar.slider("Stroke width: ", 1, 25, 3)
    #rect_height = st.slider(label='Height', min_value=10,max_value=150, value=75, step = 10)
    #rect_width = st.slider(label='Width', min_value=10,max_value=300, value=150, step=10)
    #repeat = st.slider(label='Repeats', min_value=1,max_value=100, value=5, key='rep_slider')

    x = np.zeros(5)

    x[0] = float(col1.text_input("Width 1 (ft)", value = '5'))
    #x1 = col1.slider(label='Width 1 (ft)', min_value=1,max_value=10, value=5, step = 1)
    if n_h > 1:
        x[1] = float(col1.text_input("Width 2 (ft)", value = '5'))
        #x2 = col2.slider(label='Width 2 (ft)', min_value=1,max_value=10, value=5, step = 1)
    if n_h > 2:
        x[2] = float(col1.text_input("Width 3 (ft)", value = '5'))
        #x3 = col3.slider(label='Width 3 (ft)', min_value=1,max_value=10, value=5, step = 1)
    if n_h > 3:
        x[3] = float(col1.text_input("Width 4 (ft)", value = '5'))
        #x4 = col4.slider(label='Width 4 (ft)', min_value=1,max_value=10, value=5, step = 1)
    if n_h > 4:
        x[4] = float(col1.text_input("Width 5 (ft)", value = '5'))
        #x5 = col5.slider(label='Width 5 (ft)', min_value=1,max_value=10, value=5, step = 1)

    #---Height of openings [ft]---
    #y = np.array([3, 2, 6, 1])
    y = np.zeros(5)

    y[0] = float(col2.text_input("Height 1 (ft)", value = '8'))
    #y1 = col1.slider(label='Height 1 (ft)', min_value=1,max_value=20, value=8, step = 1)
    if n_v > 1:
        y[1] = float(col2.text_input("Height 2 (ft)", value = '4'))
        #y2 = col2.slider(label='Height 2 (ft)', min_value=1,max_value=20, value=3, step = 1)
    if n_v > 2:
        y[2] = float(col2.text_input("Height 3 (ft)", value = '2'))
        #y3 = col3.slider(label='Height 3 (ft)', min_value=1,max_value=20, value=3, step = 1)
    if n_v > 3:
        y[3] = float(col2.text_input("Height 4 (ft)", value = '2'))
        #y4 = col4.slider(label='Height 4 (ft)', min_value=1,max_value=10, value=3, step = 1)
    if n_v > 4:
        y[4] = float(col2.text_input("Height 5 (ft)", value = '2'))
        #y5 = col5.slider(label='Height 5 (ft)', min_value=1,max_value=10, value=3, step = 1)



#---Horizontal repeats---
rep_h = 1
#---Vertical repeats---
rep_v = 1
#---Height of spandrel opening [ft]---
ys = 0
#---pixels per ft---
px_cal = 12*2
#x_px = x*px_cal
#y_px = y*px_cal
#---Horizontal openings---
#n_h = len(x)
#---Vertical openings---
#n_v = len(y)
#---Spandrels---
if ys > 0:
    s = 1
else:
    s = 0
#---Frame width [ft]---
w = 2.5/12
#---Vertical frame mass [lb/ft]---
#Mv = 2.3
#---Horizontal frame mass [lb/ft]---
#Mh = 2.7
#---GWP of frame [lbCO2/lb frame]---
#GWP_frame =  18
#---GWP of IGU [lbCO2/ft2 of IGU]---
#GWP_igu = 15
#---GWP of spandrel [lbCO2/ft2 of spandrel]---
#GWP_span = 30
#---Area density of IGU [lb/ft²]---
#igu_dens = 20

area_igu = 0
area_total = 0
for i in x:
    for j in y:
        area_igu += (i - w) * (j - w) * (0.3048) * (0.3048)
        area_total += i * j * (0.3048) * (0.3048)
area_igu = area_igu * rep_h * rep_v

area_span = 0
if ys > 0:
    for i in x:
        area_span += (ys - w) * (i - w) * (0.3048) * (0.3048)
area_span = area_span * rep_h * rep_v 

len_v_single = (np.sum(y) + ys) * 0.3048
len_h_single = (np.sum(x)-w*n_h) * 0.3048

len_v = len_v_single*n_h
len_h = len_h_single*(n_v + s)

len_v_total = rep_v*(len_v*rep_h + len_v_single)
len_h_total = rep_h*(len_h*rep_v + len_h_single)


#---IGU GWP (kgCO2/kg)---
igu_gwp1 = 1.26
igu_gwp2 = 1.26
   
if st.session_state.glass1 == 'Triple pane':
    #---IGU density (kg/m²)---
    igu_dens1 = 46.1
else: 
    igu_dens1 = 30.7
if st.session_state.glass2 == 'Triple pane':
    igu_dens2 = 46.1
else:
    igu_dens2 = 30.7
    
if st.session_state.man_ec1:
    em_carbon1 = float(st.session_state.ec1)
else:
    frame_dens1 = float(dfP['kg/m'].loc[(dfP['Product name']==st.session_state.prod1)])
    #frame_gwp1 = float(dfP['kgCO2/kg'].loc[(dfP['Product name']==st.session_state.prod1)])
    frame_gwp1 = 7.0 #float(dfP['2022'].loc[(dfP['Product name']==st.session_state.prod1)])
    
if st.session_state.win2:
    if st.session_state.man_ec2:
        em_carbon2 = float(st.session_state.ec2)
    else:
        frame_dens2 = float(dfP['kg/m'].loc[(dfP['Product name']==st.session_state.prod2)])
        #frame_gwp2 = float(dfP['kgCO2/kg'].loc[(dfP['Product name']==st.session_state.prod2)])
        frame_gwp2 = 7.0 #float(dfP['2022'].loc[(dfP['Product name']==st.session_state.prod2)])
        

Mv1 = frame_dens1
Mh1 = frame_dens1
Mv2 = frame_dens2
Mh2 = frame_dens2

mass_frame_v1 = len_v_total*Mv1 
mass_frame_h1 = len_h_total*Mh1

mass_frame_total1 = mass_frame_h1 + mass_frame_v1
mass_igu_total1 = igu_dens1 * area_igu
co2_igu1 = igu_gwp1 * mass_igu_total1
co2_frame1 = frame_gwp1 * mass_frame_total1
#co2_spandrel = span_gwp * area_span
co2_total1 = co2_frame1 + co2_igu1 #+ co2_spandrel
em_carbon1 = co2_total1 / area_total

# Specify canvas parameters in application
if st.session_state.win2:
    mass_frame_v2 = len_v_total*Mv2 
    mass_frame_h2 = len_h_total*Mh2
    mass_frame_total2 = mass_frame_h2 + mass_frame_v2
    mass_igu_total2 = igu_dens2 * area_igu
    co2_igu2 = igu_gwp2 * mass_igu_total2
    co2_frame2 = frame_gwp2 * mass_frame_total2
    co2_total2 = co2_frame2 + co2_igu2
    em_carbon2 = co2_total2 / area_total

y_px = np.flip(y[y!=0])*px_cal
x_px = x[x!=0]*px_cal
x_pos = 0

for i in range(len(x_px)):
    y_pos = 0
    for j in range(len(y_px)):
        if i==0 and j==0:
            init_rect = make_json(x_px[i],y_px[j],x_pos,y_pos,stroke_width,False)  
        else:
            init_rect["objects"].append(make_json(x_px[i],y_px[j],x_pos,y_pos,stroke_width,True))
        
        y_pos += y_px[j]
    x_pos += x_px[i]
canvas_height = y_pos+stroke_width #1.1*np.sum(y)
canvas_width = x_pos+stroke_width #1.1*np.sum(x)
#init_rect = make_json(50,100,0,0,5,False)
#init_rect["objects"].append(make_json(100,150,150,50,5,True))

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(155, 155, 255, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color= "#8C8C8C",
    background_color= "#FFFFFF",
    background_image= None,
    update_streamlit=False,
    height=400,#canvas_height,
    width=800,#canvas_width,
    drawing_mode="rect",
    initial_drawing=init_rect,
    point_display_radius=0,
    display_toolbar= False,
    key="canvas")


# Do something interesting with the image data and paths
#if canvas_result.image_data is not None:
    #st.image(canvas_result.image_data)
# if canvas_result.json_data is not None:
#     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     st.dataframe(objects)

st.header('PRODUCT IMPACTS')
cola, col1, col2, col3, col4, col5 = st.columns((1, 1, 1, 1, 1, 1))
    #col1.subheader('ENERGY')
    #col2.subheader('CARBON')
    #col3.subheader('FINANCIAL')
#cola.text("")
cola.subheader(str(st.session_state.prod1))
col1.metric('Embodied carbon (kgCO2/m²)',"{:,}".format(int(em_carbon1)))
col2.metric('Framing GWP (kgCO2)',"{:,}".format(int(co2_frame1)))
col3.metric('Glazing GWP (kgCO2)',"{:,}".format(int(co2_igu1)))
col4.metric('Framing mass (kg)',"{:,}".format(int(mass_frame_total1)))
col5.metric('Glazing mass (kg)',"{:,}".format(int(mass_igu_total1)))
col1.text("")
col2.text("")
col3.text("")
col4.text("")
col5.text("")
if st.session_state.win2:
    #cola.text("")
    cola.subheader(str(st.session_state.prod2))
    col1.metric('Embodied carbon (kgCO2/m²)',"{:,}".format(int(em_carbon2)),"{:.0f}".format(100*(em_carbon2-em_carbon1)/em_carbon1)+"%")
    col2.metric('Framing GWP (kgCO2)',"{:,}".format(int(co2_frame2)),"{:.0f}".format(100*(co2_frame2-co2_frame1)/co2_frame1)+"%")
    col3.metric('Glazing GWP (kgCO2)',"{:,}".format(int(co2_igu2)),"{:.0f}".format(100*(co2_igu2-co2_igu1)/co2_igu1)+"%")
    col4.metric('Framing mass (kg)',"{:,}".format(int(mass_frame_total2)),"{:.0f}".format(100*(mass_frame_total2-mass_frame_total1)/mass_frame_total1)+"%")
    col5.metric('Glazing mass (kg)',"{:,}".format(int(mass_igu_total2)),"{:.0f}".format(100*(mass_igu_total2-mass_igu_total1)/mass_igu_total1)+"%")


st.text("")
st.header('CARBON GRAPHS')
st.text("")
with st.expander(label='', expanded = False):
    col1edge, col1, col2edge = st.columns((1, 4, 1))
    with col1:
        #st.subheader('Carbon Footprint Breakdown')
        #st.text('')
        pie_data1 = np.array([co2_frame1,co2_igu1])
        colors = ['#69A761', '#617165','#85A993','#8C9078']
        labels1 = ["Frame CO2", "Glazing CO2"]
        #col1edge, col1, colmid, col2, col2edge = st.columns((1, 3, 1, 3, 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (5,5), dpi = 300)
        #ax1(figsize = (2,2))
        #ax1 = plt.subplot2grid((1,2),(0,0))
        ax1.pie(pie_data1, autopct = '%1.0f%%', colors = colors, textprops={'fontsize': 6})
        ax1.set_title(label=str(st.session_state.prod1), fontsize=8)
        #st.pyplot(fig)
        ax1.legend(labels1, loc = 'upper left', fontsize = 4)
        
        #ax2(figsize = (2,2))
        pie_data2 = np.array([co2_frame2,co2_igu2])
        #ax1 = plt.subplot2grid((2, 4), (0, 1))
        #labels2 = ["Electricity", "Natural Gas"]
        ax2.pie(pie_data2, autopct = '%1.0f%%', colors = colors, textprops={'fontsize': 6})
        ax2.set_title(label=str(st.session_state.prod2), fontsize=10)
        ax2.legend(labels1, loc = 'upper right', fontsize = 4)
        st.pyplot(fig)

        # t = np.zeros((round(yr+1),1))
        # c1 = np.zeros((round(yr+1),1))
        # c2 = np.zeros((round(yr+1),1))
        # for i in range(yr+1):
        #     t[i] = i
        #     if i == 1:
        #         c1[i] = em_carbon1
        #         c2[i] = em_carbon2
        #     elif i > 1:
        #         c1[i] = c1[i-1] + op_annual1
        #         c2[i] = c2[i-1] + op_annual2

        # fig3, ax3 = plt.subplots(figsize = (10,5), dpi = 300)
        # fs3 = 12
        # labels3 = [st.session_state.prod1, st.session_state.prod2]
        # ax3.plot(t, c1, linewidth=2.0, color = colors[0])
        # ax3.plot(t, c2, linewidth=2.0, color = colors[1])
        # ax3.set_xlabel('Time (yrs)', fontsize=fs3)
        # ax3.set_ylabel('Cumulative carbon (kgCO2/m²)', fontsize=fs3)
        # ax3.legend(labels3, fontsize=(fs3-2))
        # #ax1.set_ylabel(ylabel='Carbon Emissions (kgCO2/m²)',fontsize=fs)
        # ax3.tick_params(axis='both', which='major', labelsize=fs3)
        # #ax3.tick_params(axis='y', which='major', labelsize=fs)
        
        # #ax3.set(xlim=(0, 8), xticks=np.arange(1, 8),
        # #ylim=(0, 8), yticks=np.arange(1, 8))
        # st.pyplot(fig3)

        st.text("")
        st.text("")
        #st.dataframe(df) 

col1, col2, col3, col4, col5 = st.columns((1, 1, 1, 1, 1))
col1.metric('Total Frame Length (m)',"{:0.1f}".format(len_h_total+len_v_total))
col2.metric('Vertical Frame Length (m)',"{:0.1f}".format(len_v_total))
col3.metric('Horizontal Frame Length (m)',"{:0.1f}".format(len_h_total))
