#%%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# %%

#GOAL: To determine reasonable relationships to classify diamond forming fluids using the elements that can be measured in twinning wisps by LA-ICP-MS 

# Plots of Majors, Fluid Mobile, High Field Strength and And Melt compatible Elements Ree Elements 
#%%
path = "Weiss_et_al-RiMG-Global_Diamond_HDF_compilation_Aug_2021.xlsx"
sheet = "HDF_Limited_Col"


df = pd.read_excel(io=path,sheet_name=sheet)

fix, ax = plt.subplots()

sns.scatterplot(df, y = "Zr/La", x= "Al/La", hue = "HDF Type (calculated)", ax = ax)
# ax.set_xbound(0,.00x0002)
# ax.set_ybound(0,20)
plt.yscale('log')
plt.xscale('log')


#%%
# Ba/Ce Vs LA/Ce is pretty good for differentiating Saline Fluids as is La/Ce Pb/U
# Th/U Is also Pretty useful for this
fig, ax = plt.subplots()
sns.scatterplot(df, y = "La/Ce", x= "Ba/Ce", hue = "HDF Type (calculated)", ax = ax)
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


# %% Build log-ratio features from element concentrations
# log(element/Ce) from the concentration columns, missing kept as NaN (see hdf_compilation.py:
# the Excel ratio columns store a missing numerator as 0, and log(x + 1) squashes ratios below 1)
from hdf_compilation import HDF_TYPES, LABEL, has_trace_data, log_ratios, make_classifier

features = log_ratios(df)
has_data = df[LABEL].notna() & has_trace_data(features)
X = features[has_data].to_numpy()
y = df.loc[has_data, LABEL].to_numpy()
groups = df.loc[has_data, "Reference"].fillna("unknown").to_numpy()
features[has_data].describe()

#%% Clustering Algorithms
from sklearn import mixture
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Missing values are filled with the column median, then each log ratio is scaled to unit variance
X_scaled = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(X)

n_components = 6
gmm = mixture.GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=1,
    )

gmm.fit(X_scaled)
y_pred = gmm.predict(X_scaled)

plot_cols = ["Rb/Sr", "Zr/La", "Ba/Ce", "La/Ce"]
Classification_Comp = pd.DataFrame({"HDF_type": y, "Clusters": y_pred}, index=df.index[has_data])
Classification_Comp = Classification_Comp.join(df[plot_cols].where(df[plot_cols] > 0))
# %%
cmap = sns.color_palette("hls", n_components)
fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Rb/Sr", x= "Zr/La", hue = "Clusters", ax = ax, palette= cmap)

plt.yscale('log')
plt.xscale('log')



fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Rb/Sr", x= "Zr/La", hue = "HDF_type", ax = ax)

plt.yscale('log')
plt.xscale('log')

#%%
fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Ba/Ce", x= "La/Ce", hue = "Clusters", ax = ax, palette=cmap)

plt.yscale('log')
plt.xscale('log')

fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Ba/Ce", x= "La/Ce", hue = "HDF_type", ax = ax)

plt.yscale('log')
plt.xscale('log')
# %%
plot_order = HDF_TYPES
sns.catplot(Classification_Comp, x  = "HDF_type", y = "Clusters", order=plot_order , alpha = 0.2, size = 10 )
plt.xticks(rotation=30)

pivot_table = Classification_Comp.pivot_table(index='HDF_type', columns='Clusters', aggfunc='size', fill_value=0)
pivot_table
# %%

# Summary of classification methods 1/14/25 
# I can reliably cluster the Saline fluid inclusions from the Silic and Carbonatitic
# The Other inclusions are harder to distinguish from each other, There appear to be mixing arrays between Sili 

# Linear Trees Model doesnt work well for multi-classification problems, but will likely work for soemething like a thermometer or time series 
#%% Classification
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

# Balanced logistic regression on the log ratios (pipeline in hdf_compilation.make_classifier).
# An anisotropic-RBF Gaussian process classifier, used before 10/7/25, predicted high-Mg carbonatitic
# for almost every row (balanced accuracy 0.23-0.28, chance is 0.20).
classifier = make_classifier()

# Stratified CV mixes rows from the same paper (one lab, usually one locality) into train and test.
# Group-by-paper CV holds out whole papers: the honest estimate for classifying new data from our lab.
y_pred_strat = cross_val_predict(classifier, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
y_pred_group = cross_val_predict(classifier, X, y, cv=GroupKFold(5), groups=groups)
print(f"n = {len(y)}, chance balanced accuracy = {1 / len(plot_order):.2f}")
print(f"balanced accuracy, stratified 5-fold:     {balanced_accuracy_score(y, y_pred_strat):.3f}")
print(f"balanced accuracy, group-by-paper 5-fold: {balanced_accuracy_score(y, y_pred_group):.3f}")

#%% Group-by-paper confusion matrix: rows are the actual type, columns the predicted type
test_pivot_table = pd.DataFrame(
    confusion_matrix(y, y_pred_group, labels=plot_order), index=plot_order, columns=plot_order
)
test_pivot_table

#%% Which log ratios push a fluid toward each type (fit on all rows, coefficients in standard deviations)
classifier.fit(X, y)
coefficients = pd.DataFrame(
    classifier[-1].coef_, index=classifier[-1].classes_, columns=features.columns
).loc[plot_order]
coefficients.round(2)
# %%


y_all_pred = classifier.predict(X)
#%%
fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Ba/Ce", x= "La/Ce", hue = y_all_pred, hue_order=plot_order, ax = ax)

plt.yscale('log')
plt.xscale('log')

fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Ba/Ce", x= "La/Ce", hue = "HDF_type", hue_order=plot_order, ax = ax)

plt.yscale('log')
plt.xscale('log')
# %%


# Summary 10/7/25: regularised logistic regression on log(element/Ce) ratios reaches a balanced accuracy
# of ~0.58 (stratified) and ~0.55 (held-out papers) over the 5 types, against 0.20 for chance.
# Saline and silicic separate best; silicic - low-Mg carbonatitic is mostly confused with its neighbours
# (silicic and high-Mg carbonatitic), as expected for cuts through a mixing array.
# Predicting SiO2, MgO, CaO and Cl and applying the Weiss et al. thresholds did worse (~0.39).
# %%
