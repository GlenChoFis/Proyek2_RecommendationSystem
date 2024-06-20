import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt 
from statistics import mean
import seaborn as sns
from sklearn.cluster import KMeans

# Membaca dataset
df = pd.read_excel("data_raket_bulutangkis.xlsx")
print(df)
print("============================================================")

data=df[['Brand', 'Type', 'Color', 'Grip', 'Weight', 'Balance', 'Flexibility', 'Player_Level', 'Player_Style', 'Price']]

# Mengubah tipe data dan dikategorikan
df['Brand'] = df['Brand'].astype('category').cat.set_categories(['Yonex', 'Ashaway', 'Babolat', 'Carlton', 'Dunlop', 'Karakal', 'Li-Ning', 'FZ Forza'])
df['Grip'] = df['Grip'].astype('category').cat.set_categories(['G2', 'G3', 'G4', 'G5', 'G6'])
df['Balance'] = df['Balance'].astype('category').cat.set_categories(['Head Heavy', 'Even Balanced', 'Head Light'])
df['Flexibility'] = df['Flexibility'].astype('category').cat.set_categories(['Stiff', 'Flexible', 'Medium'])
df['Player_Level'] = df['Player_Level'].astype('category').cat.set_categories(['Beginner', 'Intermediate', 'Advanced'])
df['Player_Style'] = df['Player_Style'].astype('category').cat.set_categories(['Offensive', 'Defensive', 'All Round', 'Power', 'Speed'])
print(df)
print("============================================================")

# # Menghapus nilai NaN
# df = df.dropna(axis=1)

#mendeteksi missing value
print("Deteksi Missing Value")
print(data.isna().sum())
print("============================================================")

#menghapus data mising value
print("Penanganan Missing Value")
Missing_Brand = data['Brand'].isna().dropna()
Missing_Type = data['Type'].isna().dropna()
Missing_Color = data['Color'].isna().dropna()
Missing_Weight = data['Weight'].isna().dropna()
Missing_Grip = data['Grip'].isna().dropna()
Missing_Balance = data['Balance'].isna().dropna()
Missing_Flexibility = data['Flexibility'].isna().dropna()
Missing_Player_Level = data['Player_Level'].isna().dropna()
Missing_Player_Style = data['Player_Style'].isna().dropna()
Missing_Price = data['Price'].isna().dropna()
print("Missing_Brand = ", Missing_Brand.isna().sum())
print("Missing_Type = ", Missing_Type.isna().sum())
print("Missing_Color = ", Missing_Color.isna().sum())
print("Missing_Weight = ", Missing_Weight.isna().sum())
print("Missing_Grip = ", Missing_Grip.isna().sum())
print("Missing_Balance = ", Missing_Balance.isna().sum())
print("Missing_Flexibility = ", Missing_Flexibility.isna().sum())
print("Missing_Player_Level = ", Missing_Player_Level.isna().sum())
print("Missing_Player_Style = ", Missing_Player_Style.isna().sum())
print("Missing_Price = ", Missing_Price.isna().sum())
print("============================================================")

# Mengubah data kategorikal menjadi numerik
print("Mengubah data menjadi numerik")
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])
print(data)
print("============================================================")

#  Mendeteksi Outlier
print("Deteksi Outlier")
outlier=[]
def detect_outlier(data):
    threshold=2
    mean = np.mean(data)
    std = np.std(data)
    
    for x in data:
        z_score = (x-mean)/std
        if np.abs(z_score)>threshold:
            outlier.append(x)
    return outlier

#Mencetak outlier
outlier1 = detect_outlier(data['Brand'])
print("outlier kolom Brand: ",outlier1)
print("banyak outlier Brand: ",len(outlier1))
print()

outlier2 = detect_outlier(data['Type'])
print("outlier kolom Type : ",outlier2)
print("banyak outlier Type : ",len(outlier2))
print()

outlier3 = detect_outlier(data['Color'])
print("outlier kolom Color : ",outlier3)
print("banyak outlier Color : ",len(outlier3))
print()

outlier4 = detect_outlier(data['Grip'])
print("outlier kolom Grip: ",outlier4)
print("banyak outlier Grip: ",len(outlier4))
print()

outlier5 = detect_outlier(data['Weight'])
print("outlier kolom Weight : ",outlier5)
print("banyak outlier Weight : ",len(outlier5))
print()

outlier6 = detect_outlier(data['Balance'])
print("outlier kolom Balance : ",outlier6)
print("banyak outlier Balance : ",len(outlier6))
print()

outlier7 = detect_outlier(data['Flexibility'])
print("outlier kolom Flexibility: ",outlier7)
print("banyak outlier Flexibility: ",len(outlier7))
print()

outlier8 = detect_outlier(data['Player_Level'])
print("outlier kolom Player_Level : ",outlier8)
print("banyak outlier Player_Level : ",len(outlier8))
print()

outlier9 = detect_outlier(data['Player_Style'])
print("outlier kolom Player_Style : ",outlier9)
print("banyak outlier Player_Style : ",len(outlier9))
print()

outlier10 = detect_outlier(data['Price'])
print("outlier kolom Price : ",outlier10)
print("banyak outlier Price : ",len(outlier10))
print()
print("============================================================")

# Mengubah outlier menjadi median
def replace_outlier_with_median(data, column):
    threshold = 3
    median = np.median(data[column])

    for x in data[column]:
        z_score = (x - np.mean(data[column])) / np.std(data[column])
        if np.abs(z_score) > threshold:
            data[column] = data[column].replace(x, median)
    return data
           
# Menerapkan fungsi replace_outlier_with_median untuk setiap kolom
variabel = ['Brand', 'Type', 'Color', 'Grip', 'Weight', 'Balance', 'Flexibility', 'Player_Level', 'Player_Style', 'Price']
for var in variabel:
    data = replace_outlier_with_median(data, var)
print("============================================================")

# Memisahkan fitur dan target
x = data.drop('Player_Level', axis=1)
y = data['Player_Level']
print("data variabel")
print(x)
print("data kelas")
print(y)
print("============================================================")

#normalisasi
#feature scaling or standardization
scaler = StandardScaler()
Normalisasi = scaler.fit_transform(data)
print("Hasil Feature Scaling = ")
print(Normalisasi)
print("============================================================")

# # Menentukan jumlah kluster
# jumlah_kluster = 2

# # Melakukan K-means
# kmeans = KMeans(n_clusters=jumlah_kluster, random_state=0)
# df['Kluster'] = kmeans.fit_predict(Normalisasi)

# # Menampilkan hasil klustering
# print("Hasil Klustering:")
# print(df[['Brand', 'Player_Level', 'Player_Style', 'Kluster']])
# print("============================================================")

# # Visualisasi hasil klustering
# plt.scatter(df['Brand'], df['Player_Level'], c=df['Kluster'], cmap='rainbow')
# plt.title('Hasil Klustering dengan K-means')
# plt.xlabel('Brand')
# plt.ylabel('Player_Level')
# plt.show()

#cetak dataset baru yang telah dicluster
# df.to_excel("data_raket_bulutangkis_preprocessed.xlsx", index=False)

# Memisahkan dataset menjadi data training dan data uji  
print("SPLITTING DATA".center(75,"="))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=1)
print("instance variabel data training")
print(x_train)
print("instance kelas data training")
print(y_train)
print("instance variabel data testing")
print(x_test)
print("instance kelas data testing")
print(y_test)
print("============================================================")


# # Melatih model Decision Tree C4.5
clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
clf.fit(x_train, y_train)
prediksi = clf.predict(x_test)


# Visualize the decision tree using plot_tree
print('Visual Decision Tree')
plt.figure(figsize=(30, 10))
plot_tree(clf, filled=True, feature_names=['Brand', 'Type', 'Color', 'Grip', 'Weight', 'Balance', 'Flexibility', 'Player_Style', 'Price'], class_names=['Beginner', 'Intermediate', 'Advanced'])
plt.show()


# Evaluasi Model
print('Evaluasi Model:')
akurasi = accuracy_score(y_test, prediksi)
print("Akurasi: {:.2f}%".format(akurasi * 100))
precision = precision_score(y_test, prediksi, average='weighted')
print("Precision: {:.2f}%".format(precision * 100))
recall = recall_score(y_test, prediksi, average='weighted')
print("Recall: {:.2f}%".format(recall * 100))
f1 = f1_score(y_test, prediksi, average='weighted')
print("F1: {:.2f}%".format(f1 * 100))

print("Laporan Klasifikasi:")
print(classification_report(y_test, prediksi))

# print('Evaluasi Model :')
# akurasi = accuracy_score(y_test, prediksi)
# print('Akurasi : ', akurasi * 100, "%")
# precision = precision_score(y_test, prediksi)
# print('Precision : ' + str(precision))
# recall = recall_score(y_test, prediksi)
# print('Recall : ' + str(recall))
# f1 = f1_score(y_test, prediksi)
# print('F1-Score : ' + str(f1))

