import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# Конфигурация страницы
st.set_page_config(
    layout="wide", 
    page_title="Система скоринга субсидий", 
    page_icon="logo.png"
)

# --- Анимация загрузки (Splash Screen) ---
if 'splash_shown' not in st.session_state:
    st.session_state.splash_shown = True
    import base64
    import time
    
    if os.path.exists("logo.png"):
        with open("logo.png", "rb") as f:
            b64_logo_splash = base64.b64encode(f.read()).decode()
        
        # Создаем пустой контейнер-заглушку
        placeholder = st.empty()
        
        placeholder.markdown(f"""
        <style>
        /* Скрываем стандартный значок "Running..." Streamlit */
        [data-testid="stStatusWidget"] {{
            display: none !important;
        }}
        
        #splash-screen {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            width: 100%; height: 100%;
            background-color: var(--background-color);
            z-index: 9999999;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            animation: blurFadeOut 0.8s ease-in-out forwards;
            animation-delay: 2s;
        }}
        @keyframes blurFadeOut {{
            0% {{ opacity: 1; filter: blur(0px); }}
            100% {{ opacity: 0; filter: blur(10px); display: none; }}
        }}
        .splash-logo {{
            width: 220px;
            margin-bottom: 20px;
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        .splash-text {{
            font-size: 5rem;
            font-weight: 800;
            color: var(--text-color);
            letter-spacing: 12px;
            margin: 0;
            padding-left: 12px; /* Компенсация letter-spacing для идеального центрирования */
            text-align: center;
            animation: slideUp 1s ease-out;
        }}
        @keyframes slideUp {{
            0% {{ transform: translateY(30px); opacity: 0; }}
            100% {{ transform: translateY(0); opacity: 1; }}
        }}
        </style>
        <div id="splash-screen">
            <img src="data:image/png;base64,{b64_logo_splash}" class="splash-logo">
            <h1 class="splash-text">СОЭС</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Останавливаем работу сервера (все остальное не будет даже грузиться эти 2.5 секунды)
        # Это создает ту самую физическую "заглушку перед работой"
        time.sleep(2.6)
        
        # Полностью удаляем код сплеш скрина из сайта, "пропадает полностью"
        placeholder.empty()


# Настройка шрифта и стиля (чистый, профессиональный вид)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stMetric {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 8px;
        border: 1px solid var(--faded-text-10);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    h1, h2, h3 {
        font-weight: 600;
    }
    
    /* Убираем лишние отступы */
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Загрузка модели и данных
@st.cache_resource
def load_model():
    if os.path.exists("subsidies_scoring_model.joblib"):
        return joblib.load("subsidies_scoring_model.joblib")
    return None

@st.cache_data
def load_data():
    if os.path.exists("subsidies_scoring_data.csv"):
        return pd.read_csv("subsidies_scoring_data.csv")
    return pd.DataFrame()

model = load_model()
df = load_data()

# Заголовок с логотипом
import base64

if os.path.exists("logo.png"):
    with open("logo.png", "rb") as img_file:
        b64_logo = base64.b64encode(img_file.read()).decode()
        
    st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{b64_logo}" style="width: 85px; margin-right: 20px; object-fit: contain;">
            <div>
                <h1 style="margin: 0; padding: 0; line-height: 1.2;">Система оценки эффективности сельхозпроизводителей</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 1.1rem;">Объективное распределение государственных субсидий на основе данных</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.title("Система оценки эффективности сельхозпроизводителей")
    st.markdown("Объективное распределение государственных субсидий на основе данных")

if df.empty:
    st.error("Данные не найдены. Пожалуйста, запустите скрипты подготовки данных.")
    st.stop()

# --- Боковая панель ---
st.sidebar.header("Фильтры и настройки")
selected_region = st.sidebar.selectbox("Выберите область", ["Все"] + list(df['Region'].unique()))
status_filter = st.sidebar.multiselect("Статус заявки", df['Status'].unique(), default=df['Status'].unique())

filtered_df = df
if selected_region != "Все":
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]
if status_filter:
    filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)]

# Основные показатели
st.markdown("### Ключевые показатели")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Всего заявок", f"{len(filtered_df):,}")
with kpi2:
    st.metric("Общая сумма", f"{filtered_df['Amount'].sum()/1e9:.2f} млрд ₸")
with kpi3:
    st.metric("Средний балл ИИ", f"{filtered_df['Merit_Score'].mean():.1f}")
with kpi4:
    st.metric("Рост производительности", f"{filtered_df['Productivity_Growth'].mean():+1.1f}%")

st.markdown("---")

# Секция распределения бюджета
st.subheader("Оптимизация распределения бюджета")
budget_limit_mln = st.slider("Укажите доступный бюджет (млн ₸)", 50, 5000, 1000)
budget_limit = budget_limit_mln * 1e6

# Сортировка по баллу (Merit-based)
df_sorted = filtered_df.sort_values(by='Merit_Score', ascending=False)
df_sorted['Cumulative_Amount'] = df_sorted['Amount'].cumsum()
funded_df = df_sorted[df_sorted['Cumulative_Amount'] <= budget_limit]

st.info(f"Рекомендация ИИ: При данном бюджете предлагается одобрить {len(funded_df)} наиболее эффективных хозяйств.")

# Таблица результатов
st.markdown("### Список рекомендованных кандидатов")
st.dataframe(
    funded_df[['App_Number', 'Region', 'Amount', 'Merit_Score', 'Category']].rename(columns={
        'App_Number': 'Номер заявки',
        'Region': 'Область',
        'Amount': 'Сумма (₸)',
        'Merit_Score': 'Балл эффективности',
        'Category': 'Категория'
    }).head(20),
    use_container_width=True,
    hide_index=True
)

# Анализ конкретной заявки
st.markdown("---")
st.subheader("Детальный анализ заявителя")
selected_app = st.selectbox("Выберите номер заявки для проверки", funded_df['App_Number'].head(10))

if selected_app:
    app_data = funded_df[funded_df['App_Number'] == selected_app].iloc[0]
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("**Профиль хозяйства**")
        st.write(f"- Область: {app_data['Region']}")
        st.write(f"- Запрошенная сумма: {app_data['Amount']:,.0f} ₸")
        st.write(f"- Рост продукции: {app_data['Productivity_Growth']:.1f}%")
        st.write(f"- Индекс налоговой отдачи: {app_data['Tax_Return_Index']:.2f}")
        
    with c2:
        st.write("**Обоснование оценки ИИ**")
        factors = {
            "Производительность": app_data['Productivity_Growth'] * 1.5,
            "Налоги": app_data['Tax_Return_Index'] * 60,
            "Технологии": app_data['Tech_Score'] * 0.4,
            "Нарушения": -40 if app_data['Past_Violations'] == 1 else 0
        }
        f_df = pd.DataFrame(list(factors.items()), columns=['Фактор', 'Влияние'])
        fig_bar = px.bar(f_df, x='Влияние', y='Фактор', orientation='h', 
                         color='Влияние', color_continuous_scale='RdYlGn')
        fig_bar.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write("Разработано для Decentrathon 5.0")
