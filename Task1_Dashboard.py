import pandas as pd 
import numpy as np 
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Iris Dashboard",layout='centered')

st.title("🌸Iris Data Exploration🪻")
st.markdown("#### 💹Visualization Of Iris Dataset📊")


def load_data(path):
    df=pd.read_csv(path)
    return df

loaded_data=load_data("C:\\Users\\PMLS\\Downloads\\archive (10)\\Iris.csv")
st.success("✅ Data Loaded Succesfully")
loaded_data.drop(columns={"Id"},inplace=True)
st.sidebar.title("Data Exploration")

filter_data=st.sidebar.radio(" 📊 Choose Visalization Plot",['Scatter Plot For Sepal','Scatter Plot For Petal','Histogram','Bar Chart','Box Plot'])
if not st.sidebar.checkbox('Hide',True):
 if filter_data=='Scatter Plot For Sepal':
    sns.set()
    fig,ax=plt.subplots(figsize=(8, 4
                                 ))
    ax=plt.scatter(loaded_data.loc[0:49, 'SepalLengthCm'], loaded_data.loc[0:49, 'SepalWidthCm'], 
            label='Setosa', color='#1f77b4', edgecolor='black', s=60, alpha=0.8)
    ax=plt.scatter(loaded_data.loc[50:99, 'SepalLengthCm'], loaded_data.loc[50:99, 'SepalWidthCm'], 
            label='Versicolor', color='#ff7f0e', edgecolor='black', s=60, alpha=0.8)
    ax=plt.scatter(loaded_data.loc[100:149, 'SepalLengthCm'], loaded_data.loc[100:149, 'SepalWidthCm'], 
            label='Virginica', color='#2ca02c', edgecolor='black', s=60, alpha=0.8)
            
    plt.xlabel('Sepal Length (cm)', fontsize=12, fontweight='bold')
    plt.ylabel('Sepal Width (cm)', fontsize=12, fontweight='bold')
    plt.title('Sepal Length vs Sepal Width by Species', fontsize=15, fontweight='bold')
    plt.legend(title='Species', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig=fig)

 if filter_data=='Scatter Plot For Petal':
       fig,ax=plt.subplots(figsize=(8,4))
       ax=plt.scatter(loaded_data.loc[0:49, 'PetalLengthCm'], loaded_data.loc[0:49, 'PetalWidthCm'],
            label='Setosa', color='#1f77b4', edgecolor='black', s=60, alpha=0.8)
       ax=plt.scatter(loaded_data.loc[50:99, 'PetalLengthCm'], loaded_data.loc[50:99, 'PetalWidthCm'],
            label='Versicolor', color='#ff7f0e', edgecolor='black', s=60, alpha=0.8)
       ax=plt.scatter(loaded_data.loc[100:149, 'PetalLengthCm'], loaded_data.loc[100:149, 'PetalWidthCm'],
            label='Virginica', color='#2ca02c', edgecolor='black', s=60, alpha=0.8)
       plt.xlabel('Petal Length (cm)', fontsize=12, fontweight='bold')
       plt.ylabel('Petal Width (cm)', fontsize=12, fontweight='bold')
       plt.title('Petal Length vs Petal Width by Species', fontsize=15, fontweight='bold')
       plt.legend(title='Species', fontsize=10)
       plt.grid(True, linestyle='--', alpha=0.5)
       plt.tight_layout()
       st.pyplot(fig=fig)
 if filter_data=='Histogram':
 
    sns.set()
    ax1 = loaded_data.hist(
    color='#4C72B0',
    edgecolor='black',
    grid=False,
    figsize=(14,10)
    )
    fig = ax1.flatten()[0].figure
    plt.suptitle('Distribution of Iris Dataset Features', fontsize=18, fontweight='bold', color='#333333')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for axs in ax1.flatten():
       axs.set_facecolor('lightblue')
       axs.set_title(axs.get_title(),fontsize=14,color='#4C72B0')
       axs.title.set_color('#4C72B0')
       axs.set_xlabel(axs.get_xlabel(), fontsize=12)
       axs.set_ylabel(axs.get_ylabel(), fontsize=12)
    st.pyplot(fig)
 if filter_data=='Bar Chart':
     fig,ax=plt.subplots(figsize=(8,4))
     ax=loaded_data['Species'].value_counts().plot(kind='bar',edgecolor='black',width=.5)
     sns.set()
     for bars in ax.containers:
      ax.bar_label(bars,fontsize=8,fontweight='bold')
     ax.set_facecolor('lightblue')
     plt.xlabel("Species",fontsize=12,fontweight='bold')
     plt.ylabel("Count",fontsize=12,fontweight='bold')
     plt.xticks(rotation=0,fontweight='bold')
     plt.title("Bar Chart Of Species Distribution",fontweight='bold',fontsize=15)
     st.pyplot(fig)
 if filter_data=='Box Plot':
     fig,ax=plt.subplots(figsize=(12,6))
     sns.set()
     ax=sns.boxplot(data=loaded_data,palette='dark')
     plt.title('Boxplot of Iris Dataset Features', fontsize=15, fontweight='bold', color='#4C72B0')
     plt.ylabel('cm', fontsize=12, color='#333333')
     plt.yticks(fontsize=11, color='#333333')
     plt.gca().set_facecolor('whitesmoke')
     plt.xticks(color='#4C72B0',fontsize=12,fontweight='bold')
     plt.tight_layout()
     st.pyplot(fig)


