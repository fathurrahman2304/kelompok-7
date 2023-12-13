import pickle
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
# loading the trained model
pickle_in = open('model_heartdisease(randomforest).pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache_data()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope):   
 
    # Pre-processing user input 
    # Kolom Sex : ['Female' 'Male'] 
    if Sex == "Male":
        Sex = 1
    else:
        Sex = 0
    
    # Kolom ChestPainType : ['Asymptomatic' 'Atypical Angina' 'Non-Anginal Pain' 'Typical Angina']
    if ChestPainType == "Asymptomatic":
        ChestPainType = 0
    elif ChestPainType == "Atypical Angina":
        ChestPainType = 1
    elif ChestPainType == "Non-Anginal Pain":
        ChestPainType = 2
    else:
        ChestPainType = 3

    # Kolom FastingBS : ['Else' 'FastingBS > 120 mg/dl']
    if FastingBS == "Else":
        FastingBS = 0
    else:
        FastingBS = 1

    # Kolom RestingECG : ['LVH' 'Normal' 'ST']
    if RestingECG == "LVH":
        RestingECG = 0
    elif RestingECG == "Normal":
        RestingECG = 1
    elif RestingECG == "ST":
        RestingECG = 2
 
    # Kolom ExerciseAngina : ['No' 'Yes']
    if ExerciseAngina == "No":
        ExerciseAngina = 0
    else:
        ExerciseAngina = 1
    
    # Kolom ST_Slope : ['Down' 'Flat' 'Up']
    if ST_Slope == "Down":
        ST_Slope = 0
    elif ST_Slope == "Flat":
        ST_Slope = 1
    else:
        ST_Slope = 2
 
    # Making predictions 
    prediction = classifier.predict( 
        [[Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]])[0]
     
    if prediction == 0:
        pred = 'No Heart Disease'
    else:
        pred = 'Heart Disease'
    return pred
      
def dataclean(df):
    df['Sex'] = df['Sex'].map({'M':'Male', 'F':'Female'})
    df['ChestPainType'] = df['ChestPainType'].map({'TA':'Typical Angina', 'ATA':'Atypical Angina', 
                                                'NAP' : 'Non-Anginal Pain', 'ASY' : 'Asymptomatic'})
    df['FastingBS'] = df['FastingBS'].map({1: 'FastingBS > 120 mg/dl', 0: 'Else'})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 'Yes', 'N': 'No'})
    df['HeartDisease'] = df['HeartDisease'].map({1: 'Yes', 0: 'No'})
    return df

def plot_categorical_visualizations(df, categorical_columns):
    # Create subplots
    fig, axes = plt.subplots(len(categorical_columns), 2, figsize=(15, 30))
    fig.suptitle('Categorical Column Visualizations', y=1.02)

    for i, column in enumerate(categorical_columns):
        # Bar Plot
        sns.countplot(x=column, hue='HeartDisease', data=df, ax=axes[i, 0])
        axes[i, 0].set_title(f'Bar Plot for HeartDisease by {column}')

        # Add count values above the bars
        for p in axes[i, 0].patches:
            axes[i, 0].annotate(f'{round(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center', xytext=(0, 5), textcoords='offset points')

        # Pie Chart
        df[column].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[i, 1])
        axes[i, 1].set_title(f'Pie Chart for {column}')
        axes[i, 1].set_ylabel(None)

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

def plot_numeric_visualizations(df, numeric_columns):

    # Create subplots
    fig, axes = plt.subplots(len(numeric_columns), 3, figsize=(12, 20))
    fig.suptitle('Numeric Column Visualizations', y=1.02)

    for i, column in enumerate(numeric_columns):
        # Scatter Plot
        sns.scatterplot(x=column, y='HeartDisease', data=df, ax=axes[i, 0])
        axes[i, 0].set_title(f'Scatter Plot for {column}')

        # Box Plot
        sns.boxplot(x='HeartDisease', y=column, data=df, ax=axes[i, 1])
        axes[i, 1].set_title(f'Box Plot for {column}')

        # Histogram
        sns.kdeplot(data=df, x=column,hue='HeartDisease',ax=axes[i, 2])
        axes[i, 2].set_title(f'Kde Plot for {column}')

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

# Example usage:


def plot_stroke(df):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12, 5))
    ax2.pie(df['HeartDisease'].value_counts(), labels=df['HeartDisease'].value_counts().index, autopct='%1.1f%%')
    ax2.axis('equal')
    sns.countplot(x='HeartDisease',data=df, ax=ax1,color='green')
    for p in ax1.patches:
                ax1.annotate(f'{round(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()    
    st.pyplot(fig)

# this is the main function in which we define our webpage  
def main():       
    page=st.sidebar.radio("Pilih Menu Halaman",["DataSet","Visualisasi","Heart Disease Prediction"])
    df = pd.read_csv('heart.csv')
    df = dataclean(df)

    if page == "DataSet":
        html_temp = """ 
        <div style ="background-color:White;padding:13px"> 
        <h1 style ="color:black;text-align:center;">DataSet Heart Disease</h1> 
        </div> 
        """
        st.markdown(html_temp, unsafe_allow_html = True) 
        st.dataframe(df)
        st.text(df.shape)
    elif page == "Visualisasi":
        html_temp = """ 
        <div style ="background-color:White;padding:13px"> 
        <h1 style ="color:black;text-align:center;">Visualisasi</h1> 
        </div> 
        """
        st.markdown(html_temp, unsafe_allow_html = True)
        st.subheader('Count of Heart Diseases')
        plot_stroke(df) 
        selected_feature = st.selectbox('Pilih Kolom:',['Kolom Kategorik','Kolom Numerik'])
        if selected_feature == 'Kolom Kategorik':
            col_categorik = ['Sex', 'ChestPainType', 'FastingBS',
                             'RestingECG', 'ExerciseAngina', 'ST_Slope']            
            plot_categorical_visualizations(df, col_categorik)
        else:
            col_numerik = ['Age', 'RestingBP', 'Cholesterol','MaxHR', 'Oldpeak']
            plot_numeric_visualizations(df, col_numerik)


        
    elif page == "Heart Disease Prediction":
        html_temp = """ 
        <div style ="background-color:White;padding:13px"> 
        <h1 style ="color:black;text-align:center;"> Heart Disease Prediction ML App</h1> 
        </div> 
        """
        st.markdown(html_temp, unsafe_allow_html = True)
        col1,col2 = st.columns(2)
        with col1: 
            Age = st.number_input("Age",value=49) 
            Sex = st.selectbox('Gender',('Female','Male'),index=0)
            ChestPainType = st.selectbox('ChestPainType',('Asymptomatic','Atypical Angina','Non-Anginal Pain','Typical Angina'),index=2)
            RestingBP = st.number_input("RestingBP",value=160) 
            Cholesterol = st.number_input("Cholesterol",value=180) 
            FastingBS = st.selectbox('FastingBS',("FastingBS > 120 mg/dl","Else"),index=1)
        with col2:
            RestingECG = st.selectbox('RestingECG',('LVH','Normal','ST'),index=1)
            MaxHR = st.number_input("MaxHR",value=156) 
            ExerciseAngina = st.selectbox('ExerciseAngina',('No','Yes'),index=0)
            Oldpeak = st.number_input("Oldpeak",value=1.0) 
            ST_Slope = st.selectbox('ST_Slope',('Down','Flat','Up'),index=1)
        result =""
        if st.button("Predict"): 
            result = prediction(Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope) 
            st.success('Your result is {}'.format(result))
    st.caption(':blue[__Dibuat oleh kelompok 7__] :sunglasses:')      
if __name__=='__main__': 
    main()