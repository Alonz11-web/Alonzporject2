import streamlit as st
import pandas as pd
import os

# הגדרת תצורה של העמוד
st.set_page_config(page_title="פרויקט 2: מודל חיזוי פינגווינים", page_icon="🐧", layout="wide")

# מקדמי המודל הליניארי המעודכן בשני מאפיינים (מתוך מחברת הקוד)
W1 = 40.221217715430626  # flipper_length_mm
W2 = 11.231715679835007   # sex (Male=1, Female=0)
B = -3921.4750170197003   # Intercept

# תפריט העלאת קובץ בסרגל הצד (Sidebar) כדי לאפשר טעינת ה-952 שורות המלאות
st.sidebar.header("📁 טעינת הדאטה-סט המלא")
uploaded_file = st.sidebar.file_uploader("העלה את קובץ ה-CSV של הפינגווינים", type=["csv"])

@st.cache_data
def load_full_dataset(uploaded_file):
    # 1. אם העלית קובץ CSV דרך הממשק
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    # 2. אם הקובץ שמור מקומית באותה תיקייה
    elif os.path.exists("Penguin Species Prediction Dataset.csv"):
        data = pd.read_csv("Penguin Species Prediction Dataset.csv")
    else:
        # ניסיון טעינה מקישור גיטהאב רשת
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
        data = pd.read_csv(url)
        data = data.rename(columns={'bill_length_mm': 'culmen_length_mm', 'bill_depth_mm': 'culmen_depth_mm'})

    # ניקוי ערכים חסרים (מתקבלות 951/952 שורות בדאטה המלא מ-Kaggle)
    data = data.dropna(subset=['flipper_length_mm', 'sex', 'body_mass_g']).reset_index(drop=True)
    
    # התאמת עמודת המין להצגה בעברית וערך נומרי למודל
    data['sex_str'] = data['sex'].astype(str).str.capitalize()
    data['sex_hebrew'] = data['sex_str'].apply(lambda x: 'זכר' if x in ['Male', 'זכר'] else 'נקבה')
    data['sex_val'] = data['sex_str'].apply(lambda x: 1 if x in ['Male', 'זכר'] else 0)
    
    return data

df = load_full_dataset(uploaded_file)

# יצירת לשוניות ניווט
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 מחשבון חיזוי", 
    "📊 תהליך אימון ו-Loss", 
    "🎮 משחק: נצח את המודל", 
    "📁 הטבלה המלאה (Data)"
])

# ==========================================
# לשונית 1: מחשבון חיזוי
# ==========================================
with tab1:
    st.header("🔮 חיזוי משקל פינגווין (2 מאפיינים)")
    st.write("חיזוי משקל הפינגווין על סמך אורך הסנפיר ומין הפינגווין.")
    
    col1, col2 = st.columns(2)
    with col1:
        flipper_len = st.number_input("אורך סנפיר (מ\"מ):", min_value=150.0, max_value=250.0, value=195.0, step=0.5)
    with col2:
        sex_str = st.radio("מין הפינגווין:", ["נקבה", "זכר"], horizontal=True)
        sex_val = 1 if sex_str == "זכר" else 0
        
    if st.button("חשב חיזוי משקל"):
        predicted_weight = W1 * flipper_len + W2 * sex_val + B
        st.success(f"⚖️ המשקל החזוי: **{predicted_weight:.2f} גרם** ({predicted_weight / 1000:.2f} ק\"ג)")
        st.info(f"💡 **נוסחת החישוב:** $y = {W1:.2f} \\times {flipper_len} + {W2:.2f} \\times {sex_val} + ({B:.2f}) = {predicted_weight:.2f}$")

# ==========================================
# לשונית 2: הסבר מודל והשוואת Loss
# ==========================================
with tab2:
    st.header("📊 הסבר המודל והשוואת Loss בשלוש רמות")
    
    st.subheader("1. משמעות הפרמטרים של המודל")
    st.markdown(f"""
    * **$w_1$ (משקל אורך סנפיר = {W1:.2f}):** כל תוספת של 1 מ"מ לאורך הסנפיר מוסיפה כ-40.22 גרם למשקל החזוי.
    * **$w_2$ (משקל המין = {W2:.2f}):** פינגווין זכר (1) שוקל בממוצע כ-11.23 גרם יותר מנקבה (0) בעלת אורך סנפיר זהה.
    * **$b$ (הטיה / Intercept = {B:.2f}):** הקבוע של המודל לצורך כיול נקודת התחלה.
    """)
    
    st.divider()
    st.subheader("2. השוואת Loss (MSE) בשלוש רמות")
    
    loss_data = {
        "רמת המודל": ["1. Baseline (ממוצע בלבד)", "2. מודל במאפיין אחד (סנפיר)", "3. מודל בשני מאפיינים (סנפיר + מין)"],
        "מאפיינים בשימוש": ["ללא מאפיינים", "flipper_length_mm", "flipper_length_mm + sex"],
        "Loss (MSE)": ["494,667.05", "213,077.94", "213,046.98"]
    }
    st.table(loss_data)

# ==========================================
# לשונית 3: משחק נצח את המודל
# ==========================================
with tab3:
    st.header("🎮 משחק: נצח את המודל!")
    
    if 'game_round' not in st.session_state:
        st.session_state.game_round = 0
    if 'user_score' not in st.session_state:
        st.session_state.user_score = 0
    if 'model_score' not in st.session_state:
        st.session_state.model_score = 0
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    if 'current_sample' not in st.session_state:
        st.session_state.current_sample = df.sample(1).iloc[0]

    total_rounds = 5

    with st.expander("📖 חוקי המשחק"):
        st.write(f"""
        * **כל הסיבובים נדגמים אקראית מתוך הטבלה המלאה ({len(df)} שורות)!**
        * **המטרה:** לנחש את משקל הפינגווין האמיתי קרוב יותר מהמודל.
        * **ניקוד:** מי שהיה קרוב יותר מקבל 10 נקודות.
        """)

    if st.session_state.game_round >= total_rounds:
        st.balloons()
        st.subheader("🏁 המשחק הסתיים!")
        st.write(f"**ניקוד סופי שלך:** {st.session_state.user_score} נקודות")
        st.write(f"**ניקוד סופי של המודל:** {st.session_state.model_score} נקודות")
        
        if st.session_state.user_score > st.session_state.model_score:
            st.success("🏆 כל הכבוד! ניצחת את מודל ה-AI!")
        elif st.session_state.user_score < st.session_state.model_score:
            st.error("🤖 המודל ניצח הפעם! נסה שוב!")
        else:
            st.warning("🤝 תיקו!")
            
        if st.button("🔄 שחק שוב"):
            st.session_state.game_round = 0
            st.session_state.user_score = 0
            st.session_state.model_score = 0
            st.session_state.submitted = False
            st.session_state.current_sample = df.sample(1).iloc[0]
            st.rerun()

    else:
        sample = st.session_state.current_sample
        st.subheader(f"סיבוב {st.session_state.game_round + 1} מתוך {total_rounds}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("📏 אורך סנפיר", f"{sample['flipper_length_mm']} מ\"מ")
        with col_b:
            st.metric("🐧 מין", sample['sex_hebrew'])
            
        user_guess = st.number_input(
            "מה לדעתך המשקל האמיתי (בגרמים)?",
            min_value=2000.0,
            max_value=7000.0,
            value=4000.0,
            step=50.0,
            key=f"guess_round_{st.session_state.game_round}"
        )
        
        if st.button("שלח ניחוש") and not st.session_state.submitted:
            st.session_state.submitted = True
            model_pred = W1 * sample['flipper_length_mm'] + W2 * sample['sex_val'] + B
            actual = float(sample['body_mass_g'])
            
            user_err = abs(user_guess - actual)
            model_err = abs(model_pred - actual)
            
            if user_err < model_err:
                st.session_state.user_score += 10
                st.session_state.last_result = "🎉 היית קרוב יותר מהמודל! (+10 נקודות)"
            elif model_err < user_err:
                st.session_state.model_score += 10
                st.session_state.last_result = "🤖 המודל היה קרוב יותר! (+10 נקודות למודל)"
            else:
                st
