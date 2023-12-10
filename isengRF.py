import pickle
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
# loading the trained model
pickle_in = open('model_stroke(randomforest).pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache_data()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Gender,Age,Hypertension,heart_disease,glucose_avg,bmi):   
 
    # Pre-processing user input    
    if Gender == "Male":
        Gender = 1
    else:
        Gender = 0
    # patient job type: 0 - Never_worked, 1 - children, 2 - Govt_job, 3 - Self-employed, 4 - Private
    if Hypertension == "Yes":
        Hypertension = 1
    else:
        Hypertension = 0
    
    if heart_disease == "Yes":
        heart_disease = 1
    else:
        heart_disease = 0
 
    # LoanAmount = LoanAmount / 1000
 
    # Making predictions 
    prediction = classifier.predict( 
        [[Gender,Age,Hypertension,heart_disease,glucose_avg,bmi]])[0]
     
    if prediction == 0:
        pred = 'No Stroke'
    else:
        pred = 'Stroke'
    return pred
      
def dataclean(df):
    df = df[df['gender'] != 'Other']
    df = df.dropna()
    df = df.head(1000).reset_index().drop(columns=['index'])
    df['hypertension'] = df['hypertension'].map({0:'No',1:'Yes'})
    df['heart_disease'] = df['heart_disease'].map({0:'No',1:'Yes'})
    df['stroke'] = df['stroke'].map({0:'No',1:'Yes'})
    return df

# def plot_stroke_distribution(df, selected_feature):
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # Create a count plot for stroke distribution based on the selected feature
#     if selected_feature == 'age' or selected_feature == 'avg_glucose_level' or selected_feature == 'bmi':
#         sns.histplot(data=df, x=selected_feature, kde=True, hue='stroke', ax=ax)
#         ax.set_title(f'Stroke Distribution by {selected_feature.capitalize()}')
#         # ax.legend(['Stroke', 'No Stroke'])

#     else:
#         sns.countplot(x=selected_feature, hue='stroke', data=df, ax=ax)
#         ax.set_title(f'Stroke Distribution by {selected_feature.capitalize()}')
#         # ax.legend(['No Stroke', 'Stroke'])

#     # Display the visualization
#     st.pyplot(fig)
def plot_categorical_visualizations(df, categorical_columns):
    # Create subplots
    fig, axes = plt.subplots(len(categorical_columns), 2, figsize=(12, 24))
    fig.suptitle('Categorical Column Visualizations', y=1.02)

    for i, column in enumerate(categorical_columns):
        # Bar Plot
        sns.countplot(x=column, hue='stroke',data=df, ax=axes[i, 0])
        axes[i, 0].set_title(f'Bar Plot for stroke by {column.capitalize()}')

        # Pie Chart
        df[column].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[i, 1])
        axes[i, 1].set_title(f'Pie Chart for {column.capitalize()}')
        axes[i, 1].set_ylabel(None)


    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

def plot_numeric_visualizations(df, numeric_columns):

    # Create subplots
    fig, axes = plt.subplots(len(numeric_columns), 3, figsize=(12, 10))
    fig.suptitle('Numeric Column Visualizations', y=1.02)

    for i, column in enumerate(numeric_columns):
        # Scatter Plot
        sns.scatterplot(x=column, y='stroke', data=df, ax=axes[i, 0])
        axes[i, 0].set_title(f'Scatter Plot for {column}')

        # Box Plot
        sns.boxplot(x='stroke', y=column, data=df, ax=axes[i, 1])
        axes[i, 1].set_title(f'Box Plot for {column}')

        # Histogram
        # sns.kdeplot(x=column, hue='stroke', data=df, multiple='stack', ax=axes[i,2])
        sns.histplot(data=df, x=column, kde=True, hue='stroke', multiple='layer', ax=axes[i, 2])
        axes[i, 2].set_title(f'Histogram for {column}')

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

# Example usage:


def plot_stroke(df):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12, 5))
    ax2.pie(df['stroke'].value_counts(), labels=['No Stroke', 'Stroke'] , autopct='%1.1f%%')
    ax2.axis('equal')

    sns.countplot(x='stroke',hue='stroke', data=df, ax=ax1)
    # ax1.legend(['No Stroke', 'Stroke'])

    st.pyplot(fig)

# this is the main function in which we define our webpage  
def main():       
    page=st.sidebar.radio("Pilih Menu Halaman",["DataSet","Visualisasi","Stroke Prediction"])
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df = dataclean(df)

    if page == "DataSet":
        html_temp = """ 
        <div style ="background-color:White;padding:13px"> 
        <h1 style ="color:black;text-align:center;">DataSet Stroke</h1> 
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
        st.subheader('Count of Stroke')
        # df = dataclean(df)
        plot_stroke(df) 
        selected_feature = st.selectbox('Pilih Kolom:',['Kolom Kategorik','Kolom Numerik'])
        if selected_feature == 'Kolom Kategorik':
            kolom_categoric = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            plot_categorical_visualizations(df, kolom_categoric)
        else:
            kolom_numeric = ['age', 'avg_glucose_level', 'bmi']
            plot_numeric_visualizations(df, kolom_numeric)


        
    elif page == "Stroke Prediction":
        html_temp = """ 
        <div style ="background-color:White;padding:13px"> 
        <h1 style ="color:black;text-align:center;"> Stroke Prediction ML App</h1> 
        </div> 
        """
        
        st.markdown(html_temp, unsafe_allow_html = True) 
        Gender = st.selectbox('Gender',("Male","Female"),index=0)
        Age = st.number_input("Age",value=67) 
        Hypertension = st.selectbox('Hypertention',("Yes","No"),index=1)
        heart_disease = st.selectbox('heart_disease',("Yes","No"),index=0)
        glucose_avg = st.number_input("glucose",value=228.69) 
        bmi = st.number_input("bmi",value=36.6) 
        result =""
        if st.button("Predict"): 
            result = prediction(Gender,Age,Hypertension,heart_disease,glucose_avg,bmi) 
            st.success('Your result is {}'.format(result))
    st.caption(':blue[__Dibuat oleh kelompok 7__] :sunglasses:')      
if __name__=='__main__': 
    main()