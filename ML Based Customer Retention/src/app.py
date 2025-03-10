import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(email, notif):
    try:
        sender_email = "21b01a05d8@svecw.edu.in"
        sender_password = "lqsb ggfk vcdc nydl"
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = 'We Missed You!'
        
        body = f"""
        {notif}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)
    
def generate_notification(row, df, risk='low'):
    category = row['PreferedOrderCat']
    
    if row['CouponUsed'] > df['CouponUsed'].quantile(0.25) and risk=='low':
        return f"We've got you covered! Here are some exclusive coupons for your favorite {category}."
    
    elif row['SatisfactionScore'] < df['SatisfactionScore'].quantile(0.65) and risk=='high':
        return "Sorry to hear you're not satisfied. Can you please share your feedback with us? We're here to listen and improve."
    
    elif row['OrderCount'] > df['OrderCount'].quantile(0.25) and risk=='high':
        return "Thanks for being a loyal customer! Enjoy free shipping on your next order."
    
    elif row['Tenure'] > 3 and row['OrderCount'] == 0 and risk=='high':
        return f"We miss you! Come back and explore our latest deals in {category}."
    
    elif row['CashbackAmount'] > df['CashbackAmount'].quantile(0.25) and risk=='high':
        return f"You've earned a great cashback reward! Use it on your next purchase in {category}."
    
    return f"Hey! We appreciate you. Check out new offers in {category}!"

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'risk_category' not in st.session_state:
    st.session_state.risk_category = None
if 'notification' not in st.session_state:
    st.session_state.notification = None
if 'is_manual' not in st.session_state:
    st.session_state.is_manual = False

df = pd.read_excel('E_Commerce_Dataset.xlsx', sheet_name='E Comm')
df.drop(columns=['CustomerID'], inplace=True)
df.dropna(inplace=True)

label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop(columns=['Churn'])
y = df['Churn']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, y)

y_pred = rf.predict(X_scaled)

df['Churn Risk'] = rf.predict_proba(X_scaled)[:, 1]
df['Risk Category'] = df['Churn Risk'].apply(lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.4 else 'Low'))

df_with_categories = df.copy()
df_with_categories['PreferedOrderCat'] = df['PreferedOrderCat'].map(lambda x: label_encoders['PreferedOrderCat'].inverse_transform([x])[0])

st.title("Customer Churn Analysis")

st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Go to", ["Prediction Dashboard", "Visualizations", "Customer Outreach"])

emails = [
    '21b01a05f7@svecw.edu.in',
    '21b01a05g7@svecw.edu.in',
    '21b01a05g8@svecw.edu.in',
    '21b01a05j3@svecw.edu.in'
]

if page == "Visualizations":
    st.title("Visualizations")
    
    st.subheader("Feature Importance")
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importance, y=feature_importance.index)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    st.pyplot(plt)

    st.subheader("Customer Segmentation")
    df['Churn Risk'] = rf.predict_proba(X_scaled)[:, 1]
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Churn Risk'], bins=20, kde=True)
    plt.xticks(rotation=45)
    plt.title('Customer Segmentation Based on Churn Risk')
    plt.xlabel('Churn Risk')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
    
    st.subheader("Demographic Analysis")
    gender_churn = df.groupby('Gender')['Churn'].mean() * 100
    plt.figure(figsize=(8, 5))
    sns.barplot(x=gender_churn.index, y=gender_churn.values, palette="coolwarm")
    plt.title('Percentage of Churn by Gender')
    plt.ylabel('Churn Percentage (%)')
    plt.xlabel('Gender')
    st.pyplot(plt)
    
    st.subheader("Customer Tenure Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Tenure'], bins=20, kde=True, color='blue')
    plt.title('Distribution of Customer Tenure')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    
    st.subheader("App Usage vs Churn")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Churn'], y=df['HourSpendOnApp'])
    plt.title('App Usage Hours vs Churn')
    plt.xlabel('Churn')
    plt.ylabel('Hours Spent on App')
    st.pyplot(plt)
    
    st.subheader("Coupon Usage vs Churn")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Churn'], y=df['CouponUsed'])
    plt.title('Coupon Usage vs Churn')
    plt.xlabel('Churn')
    plt.ylabel('Coupons Used')
    st.pyplot(plt)
    
    st.subheader("Satisfaction Score vs Churn")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Churn'], y=df['SatisfactionScore'])
    plt.title('Satisfaction Score vs Churn')
    plt.xlabel('Churn')
    plt.ylabel('Satisfaction Score')
    st.pyplot(plt)

    st.subheader("Customer Complaints vs Churn")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Churn'], y=df['Complain'])
    plt.title('Customer Complaints vs Churn')
    plt.xlabel('Churn')
    plt.ylabel('Number of Complaints')
    st.pyplot(plt)
    
elif page == "Prediction Dashboard":
    st.header("Customer Selection")
    
    customer_list = df.index.tolist()
    selected_customer = st.selectbox("Select a customer", customer_list)

    manual_input = st.checkbox("Manually Enter Customer Details")
    
    if manual_input != st.session_state.is_manual:
        st.session_state.prediction_made = False
        st.session_state.is_manual = manual_input
    
    if manual_input:
        user_data = {}
        for col in X.columns:
            if col in label_encoders:
                user_data[col] = st.selectbox(col, label_encoders[col].classes_)
                user_data[col] = label_encoders[col].transform([user_data[col]])[0]
            else:
                user_data[col] = st.number_input(col, value=float(df[col].mean()))
        
        if st.button("Predict"):
            user_df = pd.DataFrame([user_data])
            user_scaled = scaler.transform(user_df)
            churn_prob = rf.predict_proba(user_scaled)[0][1]
            risk_category = 'High' if churn_prob > 0.7 else ('Medium' if churn_prob > 0.4 else 'Low')
            months = int((1 - churn_prob) * 24 + 1)
            
            st.write(f"*Churn Probability: {churn_prob:.2f}*")
            st.write(f"Customer may leave in: {months} month(s)")
            st.write(f"*Risk Category: {risk_category}*")
            
            row_data = user_data.copy()
            pref_cat_encoded = row_data['PreferedOrderCat']
            row_data['PreferedOrderCat'] = label_encoders['PreferedOrderCat'].inverse_transform([int(pref_cat_encoded)])[0]
            row_series = pd.Series(row_data)

            st.session_state.risk_category = risk_category.lower()
            st.session_state.notification = generate_notification(row_series, df, risk=risk_category.lower())
            st.session_state.prediction_made = True
    else:
        selected_features = X.iloc[selected_customer]
        selected_scaled = scaler.transform([selected_features])
        churn_prob = rf.predict_proba(selected_scaled)[0][1]
        months = int((1 - churn_prob) * 24 + 1)
        risk_category = 'High' if churn_prob > 0.7 else ('Medium' if churn_prob > 0.4 else 'Low')
        
        st.write(f"*Churn Probability: {churn_prob:.2f}*")
        st.write(f"Customer may leave in: {months} month(s)")
        st.write(f"*Risk Category: {risk_category}*")
        
        pref_cat_encoded = int(selected_features['PreferedOrderCat'])
        pref_cat = label_encoders['PreferedOrderCat'].inverse_transform([pref_cat_encoded])[0]
        
        row_data = selected_features.copy()
        row_data['PreferedOrderCat'] = pref_cat
        
        st.session_state.risk_category = risk_category.lower()
        st.session_state.notification = generate_notification(row_data, df, risk=risk_category.lower())
        st.session_state.prediction_made = True

    if (manual_input and st.session_state.prediction_made) or (not manual_input):
        if st.button("Send Email Notification"):
            notif = st.session_state.notification
            
            if manual_input:
                all_success = True
                
                for email in emails:
                    success, message = send_email(email, notif)
                    if not success:
                        all_success = False
                
                if all_success:
                    st.success("Emails sent successfully!")
                else:
                    st.error("Some emails failed to send.")
            else:
                email = random.choice(emails)
                success, message = send_email(email, notif)
                
                if success:
                    st.success(f"Email sent successfully to {email}!")
                else:
                    st.error(f"Failed to send email: {message}")

elif page == "Customer Outreach":
    st.title("Customer Outreach")
    
    df_with_categories = df.copy()
    df_with_categories['PreferedOrderCat'] = df['PreferedOrderCat'].map(lambda x: label_encoders['PreferedOrderCat'].inverse_transform([x])[0])
    
    risk_groups = df_with_categories.groupby("Risk Category")
    
    for risk_level in ["High", "Medium", "Low"]:
        st.subheader(f"{risk_level} Risk Customers")
        risk_df = risk_groups.get_group(risk_level) if risk_level in risk_groups.groups else None
        
        if risk_df is not None:
            st.write(f"Total {risk_level} Risk Customers: {len(risk_df)}")
            
            if st.button(f"Send Emails to {risk_level} Risk Customers"):
                all_success = True
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (index, row) in enumerate(risk_df.iterrows()):
                    progress = int((i + 1) / len(risk_df) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {i+1}/{len(risk_df)} customers...")
                    
                    notif = generate_notification(row, df_with_categories, risk=risk_level.lower())

                    for email in emails:
                        success, message = send_email(email, notif)
                        if not success:
                            all_success = False
                
                progress_bar.empty()
                status_text.empty()
                
                if all_success:
                    st.success("Emails sent successfully!")
                else:
                    st.error("Some emails failed to send.")
        
        st.write("---")