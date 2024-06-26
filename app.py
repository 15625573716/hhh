import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap

# Load the dataset
file_path = 'C:/Users/23632/Desktop/新文章分析/新文章进一步数据/第二阶段结局.xlsx'
sheet_name = '七个变量'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Prepare the data
y = df['Adverse outcome']
X = df.drop('Adverse outcome', axis=1)

# Train-test split
random_state = 4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

# Train the model
model = XGBClassifier(n_estimators=50, random_state=random_state)
model.fit(X_train, y_train)

# Explain the model with SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Streamlit app
def predict_adverse_outcome(input_data):
    # Make predictions
    prediction = model.predict_proba(input_data)[:, 1]
    return prediction

# Streamlit UI
st.title('预测不良结局概率')

# Display form for user input
st.subheader('输入七个变量的值')
feature_names = X.columns.tolist()
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(feature, value=0.0)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

if st.button('预测'):
    # Predict adverse outcome probability
    prediction = predict_adverse_outcome(input_df)
    st.write(f"预测的不良结局概率为: {prediction[0]}")

# Optional: Display SHAP summary plot
if st.checkbox('显示SHAP值摘要图'):
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.title('SHAP Values Summary')
    st.pyplot()

