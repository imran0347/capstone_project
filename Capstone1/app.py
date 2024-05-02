from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import sklearn
import pickle

#importing code csv
a = pd.read_csv('code.csv')
a.drop(a.columns[0],axis=1,inplace=True)

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__,template_folder='template')

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    price = request.form['Price']
    freight_value = int(request.form['FreightValue'])
    payment_value = int(request.form['PaymentValue'])
    product_weight_g = int(request.form['ProductWeight'])
    product_length_cm = int(request.form['ProductLength'])
    product_height_cm = int(request.form['ProductHeight'])
    product_width_cm = int(request.form['ProductWidth'])
    product_category_name_english1 = request.form['ProductCategoryName']
    filtered_df = a[a['product_category_name_english'] == product_category_name_english1]
    encoded_values = filtered_df['encoded_column'].values
    product_category_name_english = encoded_values[0]
    feature_list = [price,freight_value,payment_value,product_weight_g,product_length_cm,product_height_cm,product_width_cm,product_category_name_english]
    single_pred = np.array(feature_list).reshape(1, -1)
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)
    crop_dict = {1: "Worse", 2: "Bad", 3: "Average", 4: "Good", 5: "Excellent"}
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "The product which you selected may comes under '{}' category".format(crop)
    else:
        result = "Sorry, we could not find the product review score."
    return render_template('index.html', result=result)

# python main
if __name__ == "__main__":
    app.run(debug=True)