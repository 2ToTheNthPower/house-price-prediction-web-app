import numpy as np
import pandas as pd
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2


def visualize_2D(dataframe, response):
    
    RESPONSE = pd.DataFrame(response)
    
    try:
        dataframe.drop(columns = ['SalePrice'], inplace = True)
    except:
        pass
    
    st.text("Scaling dataframe mean and standard deviation ...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    st.text("Done scaling dataframe.")
    
    st.text('-'*25)
    
    st.text("Projecting features onto 2-dimensional sub-space for easy visualization ...")
    pca = PCA(n_components = 2)
    data_2D = pca.fit_transform(scaled_data)
    st.text("Done projecting onto 2-dimensional sub-space.")
    
    st.text('-'*25)
    
    st.write("Total retained variance: ", pca.explained_variance_ratio_.sum()*100, "%")
    
    st.text('-'*25)
    
    st.write("See below table for % variance contribution of each feature to the two principal components (PC-1, PC-2).")
    
    st.text('-'*25)
    
    # print() statement below from https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
    
    st.write(pd.DataFrame(np.absolute(pca.components_) * 100, columns=dataframe.columns, index = ['PC-1','PC-2']))
    
    st.text('-'*25)
    st.write("""A chart of the the two principal components is shown below.  The darker the shade of the circle, 
                the larger the sale price of the hosue.  There is a clear trend that as you get further up and to 
                the right, the house prices increase.""")
    data_2D = pd.DataFrame(data_2D)
    img = data_2D.plot.scatter(x=data_2D.columns[0], y=data_2D.columns[1], c=np.array(response)/max(np.array(response)))
    st.pyplot()

def preprocessing():
    
    x_train.drop("Neighborhood", inplace=True, axis=1)
    x_test.drop("Neighborhood", inplace=True, axis=1)
    
    x_train.fillna(0, inplace=True)
    x_test.fillna(0, inplace=True)
    
    categorical_cols = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", 
                        "LandSlope", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", 
                        "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", 
                        "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", 
                        "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", 
                        "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive",
                        "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]
    
    encoder = OrdinalEncoder()
    encoder.fit(pd.concat([x_train[categorical_cols], x_test[categorical_cols]]).applymap(str).values)
    x_train[categorical_cols] = pd.DataFrame(encoder.transform(x_train[categorical_cols].applymap(str).values))
    x_test[categorical_cols] = pd.DataFrame(encoder.transform(x_test[categorical_cols].applymap(str).values))

st.title("House Pricing Assistant -- Using Artificial Intelligence")

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
desc = open("data_description.txt", "r")
desc_text = desc.read()

st.subheader("Use the window below to explore the unaltered dataset used to train the AI.")
st.dataframe(train_data)

show_key = st.checkbox('Show more information about training data')

if show_key:
    st.success(desc_text)

x_train = train_data.drop(columns=["SalePrice"])
y_train = train_data["SalePrice"]

x_test = test_data


preprocessing()

train_button = st.checkbox("Visualize Data and Train Model")
if train_button:

    visualize_2D(x_train, y_train)

    # (Next plot made with coding help from https://towardsdatascience.com/how-to-perform-exploratory-data-analysis-with-seaborn-97e3413e841d)

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    for variable, subplot in zip(train_data.drop(columns=["SalePrice","Id","LotFrontage","LotArea"]), ax.flatten()):
        sns.countplot(x_train[variable], ax=subplot)
        for label in subplot.get_xticklabels():
            label.set_rotation(90)

    st.write("Shown below is a countplot of the pre-PCA data.  As can be clearly seen, the most frequent MSSubClass is 20, the most common MSZoning value is 4, and the most common lot shape is 3.  Note that these numbers are after the text data has undergone automatic ordinal encoding.")
    st.write(fig)

    st.write("Shown next is the correlation matrix between the all of the features in the training dataset.")
    ### With help from https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas#29432741
    plt.matshow(x_train.corr())
    st.pyplot()

    ### With substantial help from https://stackoverflow.com/questions/39839112/the-easiest-way-for-getting-feature-names-after-running-selectkbest-in-scikit-le
    with st.spinner(text="Selecting top 10 most important features..."):
        feature_selection = SelectKBest(score_func=chi2, k=10)
        feature_selection.fit(x_train, y_train)
        x_train_top_ten = feature_selection.transform(x_train)
        mask = feature_selection.get_support()
        new_features = []

        feature_names = list(x_train.columns.values)

        for bool, feature in zip(mask, feature_names):
            if bool:
                new_features.append(feature)
    st.success("Top ten features selected!  See top ten features summary in window below.")
    st.write("(These are the features that will be used to train the model and make predictions.)")
    ###
            
    x_train_top_ten = pd.DataFrame(x_train_top_ten, columns=new_features)

    st.dataframe(x_train_top_ten)
    st.subheader("Please be patient during training.  The training process can take up to 10 minutes.")
    with st.spinner(text="Training predictive model on dataset..."):

        model = lgb.LGBMRegressor()

        model.fit(x_train_top_ten, y_train) 

    st.success("Done training predictive model!")

    st.header("Input data below to make predictions.")
    lotArea = st.number_input(label="Lot Area in Square Feet")
    masArea = st.number_input(label="Masonry Veneer area in Square Feet")
    basFinArea = st.number_input(label="Square feet for first basement finish style")
    basSecFinArea = st.number_input(label="Square feet for second basement finish style")
    basUnfArea = st.number_input(label="Unfinished square feet of basement area")
    secFloorArea = st.number_input(label="Area of second floor in square feet")
    lowQltyFinArea = st.number_input(label="Low quality finished square feet (all floors)")
    livingArea = st.number_input(label="Above ground living area square feet")
    poolArea = st.number_input(label="Pool area in square feet")
    miscVal = st.number_input(label="Value of miscellaneous feature (Including elevator, second garage, shed, tennis court, etc.)")

    start_predictions = st.button(label="Predict House Price")

    if start_predictions:
        st.subheader("Please be patient.  The prediction process may take up to 5 minutes.")
        predict_data = np.array([lotArea, masArea, basFinArea, basSecFinArea, basUnfArea, secFloorArea, lowQltyFinArea, livingArea, poolArea, miscVal])
        predict_data = predict_data.reshape(1, -1)
        prediction = model.predict(predict_data)
        st.success("The predicted sale price is $" + str(prediction[0].round(2)))

# Heroku deployment by following instructions at https://gilberttanner.com/blog/deploying-your-streamlit-dashboard-with-heroku