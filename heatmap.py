import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import altair as alt
import pydeck as pdk
import streamlit as st
from PIL import Image

st.set_page_config(layout="wide", page_title="Улаанбаатрын автобусний эрэлтийг шинжлэх нь", page_icon=":taxi:")


#@st.cache_resource
def load_data(x):
    #path = "combined."
    #if not os.path.isfile(path):
        #path = f"https://github.com/streamlit/demo-uber-nyc-pickups/raw/main/{path}"

    data = pd.read_csv(
        str(x)+'.csv.gz',
        #nrows=100000,  # approx. 10% of data
        names=[
            "date/time",
            "lat",
            "lon",
        ],  # specify names directly since they don't change
        skiprows=1,  # don't read header since names specified directly
        usecols=[0, 1, 2],  # doesn't load last column, constant value "B02512"
        parse_dates=[
            "date/time"
        ],  # set as datetime instead of converting after the fact
    )

    return data

def load_data_off(x):
    #path = "combined."
    #if not os.path.isfile(path):
        #path = f"https://github.com/streamlit/demo-uber-nyc-pickups/raw/main/{path}"

    data = pd.read_csv(
        str(x)+'_off.csv.gz',
        #nrows=100000,  # approx. 10% of data
        names=[
            "date/time",
            "lat",
            "lon",
        ],  # specify names directly since they don't change
        skiprows=1,  # don't read header since names specified directly
        usecols=[0, 1, 2],  # doesn't load last column, constant value "B02512"
        parse_dates=[
            "date/time"
        ],  # set as datetime instead of converting after the fact
    )

    return data



def map(data, lat, lon, zoom):
    st.write(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state={
                "latitude": lat,
                "longitude": lon,
                "zoom": zoom,
                "pitch": 50,
            },
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=data,
                    get_position=["lon", "lat"],
                    radius=100,
                    elevation_scale=4,
                    elevation_range=[0, 1000],
                    pickable=True,
                    extruded=True,
                ),
            ],
        )
    )
    
def map_off(data, lat, lon, zoom):
    st.write(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state={
                "latitude": lat,
                "longitude": lon,
                "zoom": zoom,
                "pitch": 50,
            },
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=data,
                    get_position=["lon", "lat"],
                    radius=100,
                    elevation_scale=0.8,
                    elevation_range=[0, 1000],
                    pickable=True,
                    extruded=True,
                ),
            ],
        )
    )


#@st.cache_data
def filterdata(df, hour_selected):
    return df[df["date/time"].dt.hour == hour_selected]


#@st.cache_data
def mpoint(lat, lon):
    return (np.average(lat), np.average(lon))


#@st.cache_data
def histdata(df, hr):
    filtered = data[
        (df["date/time"].dt.hour >= hr) & (df["date/time"].dt.hour < (hr + 1))
    ]

    hist = np.histogram(filtered["date/time"].dt.minute, bins=60, range=(0, 60))[0]

    return pd.DataFrame({"minute": range(60), "pickups": hist})



row1_1, row1_2 = st.columns((3, 4))


if not st.session_state.get("url_synced", False):
    try:
        pickup_hour = int(st.experimental_get_query_params()["pickup_hour"][0])
        st.session_state["pickup_hour"] = pickup_hour
        st.session_state["url_synced"] = True
    except KeyError:
        pass



def update_query_params():
    hour_selected = st.session_state["pickup_hour"]
    st.experimental_set_query_params(pickup_hour=hour_selected)
    

with row1_1:
    st.title("Улаанбаатарын нийтийн тээврийн өгөгдлийн шинжилгээ")
    
with row1_2:
    st.write(
        """
    ##
    Улаанбаатарын 3 сарын 6-ны өдрийн нийт автобусаар үйлчлүүлэгчдийн картаа даран зорчсон өгөгдлийг ашиглан энэхүү 3D heatmap-ийг боловсруулав.
    """
    )
    st.write(
             """
    LetuMongolia их сургуулийн сурагч Д.Галбадрал, DataDuran 2023.
    """
    )
    
tab1, tab2 = st.tabs(["3D heatmap", "Өгөгдлийн шинжилгээ"])

with tab1:
    row11_1, row11_2, row11_3 = st.columns((3,3,1))
    with row11_1:
        hour_selected = st.slider(
            "Цаг сонгох:", 0, 23, key="pickup_hour", on_change=update_query_params
        )
    with row11_2:
        genre = st.radio("Зорчигчийн төрөл сонгох:",('Нийт','Энгийн', 'Сурагч', 'Ахмад','Оюутан'),horizontal = True)

    choice='combined'
    if genre == 'Нийт':
        choice='combined'
    elif genre =='Энгийн':
        choice= 'normal'
    elif genre=='Сурагч':
        choice='student'
    elif genre=='Ахмад':
        choice='elder'
    elif genre=='Оюутан':
        choice='h_student'

    data = load_data(choice)
    data_off = load_data_off(choice)


    row2_1, row2_2 = st.columns((4,3))



    zoom_level = 12
    midpoint = mpoint(data["lat"], data["lon"])



    with row2_1:
        st.write(
            f"""**Автобусны буудлуудаас автобусанд суух ачаалал, {hour_selected}:00 болон {(hour_selected + 1) % 24}:00 цагийн хооронд**"""
        )
        map(filterdata(data, hour_selected), midpoint[0], midpoint[1], 11)

    with row2_2:
        st.write(
            f"""**Тус цагт автобуснаас картаа дарж буух хүмүүсийн ачаалал**"""
        )
        map_off(filterdata(data_off, hour_selected), midpoint[0], midpoint[1], 11)



    all_data=pd.read_csv("df.csv.gz")
    if genre != "Нийт":
        sorted_genre = all_data.loc[all_data['Төрөл'].isin([str(genre)])]
    else:
        sorted_genre= all_data
    sorted_time=sorted_genre.loc[sorted_genre['hour'].isin([hour_selected])]

    result = sorted_time.groupby(sorted_time['name_x']).count().reset_index().sort_values('Төрөл',ascending=False)
    result2 = sorted_time.groupby(sorted_time['Чиглэл']).count().reset_index().sort_values('Төрөл',ascending=False)
    result = result.rename(columns={'name_x': 'Автобусны буудал','Төрөл':'Эрэлт'})
    result2 = result2.rename(columns={'Төрөл':'Эрэлт'})


    chart_data = histdata(data, hour_selected)


    row4_1, row4_2,row4_3 = st.columns((3,1,4))

    with row4_1:
        st.write(
        f"""**{hour_selected}:00 цагаас {(hour_selected + 1) % 24}:00-н цагийн хооронд:**"""
        )
    row3_1, row3_2  = st.columns((4,3))

    with row3_1:
        st.write(
        f"""**Хамгийн эрэлттэй автобусны буудлууд**"""
        )
        result=result[['Автобусны буудал','Эрэлт']][:10].reset_index(drop=True)
        st.bar_chart(result,x='Автобусны буудал',y='Эрэлт',height=430)

    with row3_2:
        st.write(
        f"""**Хамгийн эрэлттэй автобусны чиглэлүүд**"""
        )
        result2=result2[['Чиглэл','Эрэлт']][:10].reset_index(drop=True)
        st.bar_chart(result2,x='Чиглэл',y='Эрэлт')
        
with tab2:
    row5_1, row5_2,row5_3 = st.columns((1,3,1))
    with row5_2:
        st.markdown("Миний хувьд ДатаДуран-д өгөгдсөн нийтийн тээврийн өгөгдлүүдээс Смарт картын гүйлгээний өгөгдлийг ашиглан Уланбаатарын нийтийн тээвэрийн ачаалал тэгш бус байгааг зургаар мөн 3D heatmap аар үзүүлэхийг зорилоо.") 
        st.markdown("Миний хувьд энэхүү шинжилгээнээс өөрийн ихэвчлэн зорчдог байсан яармагаас хотын төв хүртэлх чиглэлд автобусанд суух хүмүүсийн ачаалал их байх болов уу гэсэн хүлээлтэй байсан. Гэвч өөрийн шинжилгээнээс харахад төв зам дагуу өглөөдөө хоёр захад ойроодоо голд мөн 2 захруу ачаалал хамгийн ихтэй байгаа нь Улаанбаатарт ядаж төв зам дагасан метро маш том түгжрэлийн шийдэл болж өгөхийг харуулж байна гэж үзэж байна. Хүн бүр л өөрийгөө их түгжрээтэй хүн олонтой газар амьдрдаг гэж боддог байх гэж би боддог тул миний шинжилгээ сонирхолтой байж магадгүй юм.")
        st.header("Шинжилгээнээс гарсан график мэдээллүүд")
        st.markdown("Смарт картын өгөгдөлд 2 өдөр байсан. Үүнд эхнийх нь нэг дахь өдөр буюу жирийн ажлын өдөр 3 сарын 6-н, нөгөө нь 3 сарын 8-н буюу бүх нийтийн амралтын өдөр Мартын 8-н байсан. Heatmap-д үзүүлсэн 3 сарын 6-ны жирийн өдөр нь Улаанбаатар хотын жирийн нэгэн дундаж түгжрээтэй өдөр тул heatmap-д сонгосон болно. Мөн өгөгдсөн өгөгдөл дээр байршлийг нэмж оруулахад 20 мянган nan value өгсөн нь нийт датаны нэг хувь байсан тул heatmap-д оруулагүй ба үүнд Монгол улсын их сургуулийн буудал багтсанд хүлцэл өчий.")
        st.markdown("Цаашид баяжуулах төлөвлөгөөний хувьд би сургуулиудын байршлыг оруулан хамгийн их ачаалалтай байршилд байгаа сургуулиудыг тодруулахыг хүсэж байна. Мөн автобусны чиглэлийн зам дагуу байдлаар 2D heatmap тус цагын тус байрлалын дундаж хурдаар нь хийж өгвөл улаанбаатрын түгжрэлийг googlemap-ийн улаан шар ногоон өнгөөс илүү бодитоогоор харж чадна гэж бодож байна.Мөн илүү их датан дээр Machine learning ашиглан хүмүүсыг яаж ямар автобусаар явбал хурдан байх вэ гэдгийг тооцоолдог модел хийж болох юм.")
        image1 = Image.open('1.JPG')
        st.caption("   *Хоцрохгүйг хичээж байгаатай холбоотойгоор өглөө бол нийтийн тээврийн хамгийн их ачааллын үе.")
        st.image(image1, caption='Зураг 1')
        
        st.caption("   *Харин амралтын өдөр бол өөр хэрэг.")
        image2 = Image.open('2.jpg')
        st.image(image2, caption='Зураг 2')
        
        st.caption("   *Дараах графикаас насны онцлогийг харж болно.")
        image3 = Image.open('3.JPG')
        st.image(image3, caption='Зураг 3')
        
        st.caption("   *Ахмадууд маань мартын 8-нд арай илүү идвэхитэй байна.")
        image4 = Image.open('4.JPG')
        st.image(image4, caption='Зураг 4')
        
        image5 = Image.open('5.JPG')
        st.image(image5, caption='Зураг 5')
        st.markdown("Ч-1 нь Таван шар-Офицеруудын ордон чиглэлийн автобус бол Ч-59 нь ХМК-Офицеруудын ордон чиглэлийн тролейбус юм.")
        
        
        image6 = Image.open('6.JPG')
        st.image(image6, caption='Зураг 6')
        
        st.caption("   *Төв замын автобуснууд хамгийн их ачаалалтай байна.")
        image7= Image.open('7.JPG')
        st.image(image7, caption='Зураг 7')
        
        image8 = Image.open('8.JPG')
        st.image(image8, caption='Зураг 8')
            
        image9 = Image.open('9.JPG')
        st.image(image9, caption='Зураг 9')
        
        image10 = Image.open('10.JPG')
        st.image(image10, caption='Зураг 10')
        
        image11 = Image.open('11.JPG')
        st.image(image11 , caption='Зураг 11 ')
        
        image12  = Image.open('12.JPG')
        st.image(image12, caption='Зураг 12')
         
        st.caption("   *Смарт карт өгөгдлийн хоорондын хамаарал.")    
        image13 = Image.open('13.JPG')
        st.image(image13, caption='Зураг 13')
        
        
        st.caption("   *Автобусанд картаа дарсан хүмүүсийн харьцаа.")
        image14 = Image.open('14.JPG')
        st.image(image14, caption='Зураг 14')
        
        st.header("Анхаарал тавьсанд баярлалаа")
        

        

    