#%%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
# %%

#GOAL: To determine reasonable relationships to classify diamond forming fluids using the elements that can be measured in twinning wisps by LA-ICP-MS 

# Plots of Majors, Fluid Mobile, High Field Strength and And Melt compatible Elements Ree Elements 
#%%
path = "Weiss_et_al-RiMG-Global_Diamond_HDF_compilation_Aug_2021.xlsx"
sheet = "HDF_Limited_Col"


df = pd.read_excel(io=path,sheet_name=sheet)

fix, ax = plt.subplots()

sns.scatterplot(df, y = "Zr/La", x= "Pb/U", hue = "Craton", ax = ax)
# ax.set_xbound(0,.00x0002)
# ax.set_ybound(0,20)
plt.yscale('log')
plt.xscale('log')


#%%
# Ba/Ce Vs LA/Ce is pretty good for differentiating Saline Fluids as is La/Ce Pb/U
# Th/U Is also Pretty useful for this
fig, ax = plt.subplots()
sns.scatterplot(df, y = "La/Ce", x= "Ba/Ce", hue = "Craton", ax = ax)
# ax.set_xbound(0,.000002)

plt.yscale('log')
plt.xscale('log')
#ax.set_ybound(0.1,1)
# %%

#Mg/Al  vs Zr/La or Rb/Sr are Good at discriminating between Carbonatites and Silicic 
#Y/La or Zr/Sr
#Al/La vs Zr/La Is very good
# 

#" low ionic potential (e.g. Na, K and Rb) are generally considered to be mobile, whereas high-field-strength elements (HFSEs; e.g. Ti, Zr and Th) and REEs largely remain immobile (e.g. Pearce and Cann, 1973;Rollinson, 1993). Indeed, when several elements are plotted against Zr, the HFSEs and REEs ca"


#%%
fig, ax = plt.subplots()
sns.scatterplot(df, y = "Nb/La", x= "Zr/Sr", hue = "HDF Type (calculated)", ax = ax)
#ax.set_xbound(0,100)
# ax.set_ybound(0,20)
plt.yscale('log')
plt.xscale('log')


# %% Select Data Columns for Clustering
filtered_col = [ "Nb/La", "Zr/Sr", "Zr/La", "Rb/Sr", "Ba/Sr", "U/Sr", "Ba/Ce", "La/Ce", "Pb/U", "Y/La", "Zr/Y", "Zr/Nb", "Nb/La", 
                "MgO/La", "Al2O3/La", "MgO/Sr", "MgO/Ba", "Al2O3/Sr", "Al2O3/Ba"
                 # "Al/La", "Mg/La", "Mg/Sr"
                ]
df_filtered = df[filtered_col].dropna()
df_filtered.head()
Craton = df.loc[df_filtered.index]["Craton"]
#%% Clustering Algorithms
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

X = df_filtered.to_numpy()
y = Craton
# apply log transfrom to data for normalization
def log_transform(x):
    print(x)
    return np.log(x + 1)


scaler = StandardScaler()
transformer = FunctionTransformer(log_transform)
X = transformer.fit_transform(X)
X = StandardScaler().fit_transform(X)


#
n_components = 6
gmm = mixture.GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=1,
    )

gmm.fit(X)
y_pred = gmm.predict(X)


Classification_Comp = pd.DataFrame({"Craton":Craton, "Clusters": y_pred})
Classification_Comp = Classification_Comp.join(df_filtered)
# %%
cmap = sns.color_palette("hls", n_components)
fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Rb/Sr", x= "Zr/La", hue = "Clusters", ax = ax, palette= cmap)

plt.yscale('log')
plt.xscale('log')



fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Rb/Sr", x= "Zr/La", hue = "Craton", ax = ax)

plt.yscale('log')
plt.xscale('log')

#%%
fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Ba/Ce", x= "La/Ce", hue = "Clusters", ax = ax, palette=cmap)

plt.yscale('log')
plt.xscale('log')

fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Ba/Ce", x= "La/Ce", hue = "Craton", ax = ax)

plt.yscale('log')
plt.xscale('log')
# %%
#plot_order = ["saline", 'silicic', 'silicic - low-Mg carbonatitic', 'low-Mg carbonatitic','high-Mg carbonatitic'  ]
sns.catplot(Classification_Comp, x  = "Craton", y = "Clusters" , alpha = 0.2, size = 10 )
plt.xticks(rotation=30)

pivot_table = Classification_Comp.pivot_table(index='Craton', columns='Clusters', aggfunc='size', fill_value=0)
pivot_table
# %%

# Summary of classification methods 1/14/25 
# I can reliably cluster the Saline fluid inclusions from the Silic and Carbonatitic
# The Other inclusions are harder to distinguish from each other, There appear to be mixing arrays between Sili 

# Linear Trees Model doesnt work well for multi-classification problems, but will likely work for soemething like a thermometer or time series 
#%%


from sklearn.model_selection import train_test_split

# Assuming X is your feature data and y is your target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=134)


from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

kernel_init_array = np.ones(X.shape[1])
kernel = 1.0 * RBF(kernel_init_array )
gpc_rbf_anisotropic = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train)

# %%
y__test_pred = gpc_rbf_anisotropic.predict(X_test)

test_Classification_Comp = pd.DataFrame({"Actual":y_test, "Predicted": y__test_pred})

#%%
test_pivot_table = test_Classification_Comp.pivot_table(index='Actual', columns='Predicted', aggfunc='size', fill_value=0)
test_pivot_table

#%%
#plot_order = ["saline", 'silicic', 'silicic - low-Mg carbonatitic', 'low-Mg carbonatitic','high-Mg carbonatitic' ]
sns.catplot(test_Classification_Comp, x  = "Actual", y = "Predicted",  alpha = 0.2, size = 10)
plt.xticks(rotation=60)
plt.yticks(rotation=60)
# %%


y_all_pred = gpc_rbf_anisotropic.predict(X)
#%%
fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Ba/Ce", x= "La/Ce", hue = y_all_pred, ax = ax, palette=cmap)

plt.yscale('log')
plt.xscale('log')

fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Ba/Ce", x= "La/Ce", hue = "Craton", ax = ax)

plt.yscale('log')
plt.xscale('log')
# %%


# Using A Gaussian Mixture process with an anisotropic kernel,
#  I can classify Silicic, Carbonatitic, and Saline Fluid inclusions in Fibrous Diamonds. 
# %%
