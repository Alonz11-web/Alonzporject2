import streamlit as st
import pandas as pd
import kagglehub

st.set_page_config(page_title="פרויקט 2 - חיזוי פינגווינים", page_icon="🐧")

# מקדמי המודל מתוך מחברת הקוד
W1 = 40.22  # משקל אורך סנפיר
W2 = 11.23  # משקל מין (זכר=1, נקבה=0)
B = -3921.48  # הטיה (Intercept)

# טעינת הטבלה המלאה מהרשת
@st.cache_data
def load_data():
    url = "srisyra02/penguin-species-dataset"
    data = pd.read_csv(url).dropna(subset=['flipper_length_mm', 'sex', 'body_mass_g'])
    data['sex_val'] = data['sex'].apply(lambda x: 1 if str(x).lower() == 'male' else 0)
    data['sex_hebrew'] = data['sex'].apply(lambda x: 'זכר' if str(x).lower() == 'male' else 'נקבה')
    return data

df = load_data()

# יצירת לשוניות
tab1, tab2, tab3, tab4 = st.tabs(["🔮 חיזוי", "📊 הסבר ו-Loss", "🎮 משחק", "📁 טבלה"])

# 1. לשונית חיזוי
with tab1:
    st.header("חיזוי משקל פינגווין")
    flipper = st.number_input("אורך סנפיר (מ\"מ):", value=195.0, step=1.0)
    sex = st.radio("מין:", ["נקבה", "זכר"], horizontal=True)
    sex_val = 1 if sex == "זכר" else 0
    
    if st.button("חשב"):
        pred = W1 * flipper + W2 * sex_val + B
        st.success(f"המשקל החזוי: **{pred:.1f} גרם**")

# 2. לשונית הסבר ו-Loss
with tab2:
    st.header("הסבר המודל והשוואת Loss")
    st.write(f"• **w1 ({W1}):** כל מ\"מ בסנפיר מוסיף כ-40.22 גרם.")
    st.write(f"• **w2 ({W2}):** פינגווין זכר שוקל כ-11.23 גרם יותר מנקבה.")
    st.write(f"• **b ({B}):** קבוע הכיול.")
    
    st.subheader("השוואת Loss (MSE)")
    st.table({
        "רמה": ["1. Baseline (ממוצע)", "2. מאפיין אחד (סנפיר)", "3. שני מאפיינים (סנפיר + מין)"],
        "Loss (MSE)": ["494,667", "213,077", "213,046"]
    })

# 3. לשונית משחק
with tab3:
    st.header("משחק: נצח את המודל!")
    
    if 'round' not in st.session_state:
        st.session_state.round = 0
        st.session_state.user_score = 0
        st.session_state.model_score = 0

    if st.session_state.round < 5:
        # דגימת פינגווין לפי סיבוב
        sample = df.iloc[st.session_state.round * 15]
        
        st.write(f"### סיבוב {st.session_state.round + 1} מתוך 5")
        st.write(f"📏 אורך סנפיר: **{sample['flipper_length_mm']} מ\"מ** | 🐧 מין: **{sample['sex_hebrew']}**")
        
        guess = st.number_input("מה המשקל האמיתי (גרמים)?", value=4000.0, step=50.0, key=f"g_{st.session_state.round}")
        
        if st.button("שלח ניחוש"):
            actual = sample['body_mass_g']
            model_pred = W1 * sample['flipper_length_mm'] + W2 * sample['sex_val'] + B
            
            u_err = abs(guess - actual)
            m_err = abs(model_pred - actual)
            
            st.write(f"• **משקל אמיתי:** {actual} גרם")
            st.write(f"• **חיזוי המודל:** {model_pred:.1f} גרם (שגיאה: {m_err:.1f})")
            st.write(f"• **הניחוש שלך:** {guess} גרם (שגיאה: {u_err:.1f})")
            
            if u_err < m_err:
                st.success("🎉 ניצחת בסיבוב הזה! (+10 נקודות)")
                st.session_state.user_score += 10
            else:
                st.error("🤖 המודל ניצח בסיבוב הזה! (+10 למודל)")
                st.session_state.model_score += 10
                
            st.session_state.round += 1
            st.button("לסיבוב הבא ➡️")
            
    else:
        st.balloons()
        st.subheader("🏁 המשחק הסתיים!")
        st.write(f"**תוצאה סופית:** אתה: **{st.session_state.user_score}** | מודל: **{st.session_state.model_score}**")
        
        if st.button("🔄 שחק שוב"):
            st.session_state.round = 0
            st.session_state.user_score = 0
            st.session_state.model_score = 0
            st.rerun()

# 4. לשונית טבלה
with tab4:
    st.header("📁 הטבלה המלאה")
    st.dataframe(df[['species', 'island', 'flipper_length_mm', 'sex_hebrew', 'body_mass_g']])
