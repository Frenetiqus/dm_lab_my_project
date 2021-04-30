import pandas as pd
import numpy as np
import JSONStringConverter as converter
import MyEncoders as ec
import HelperFunctions as hf

"""Reading data from file"""
# Read data from csv file and replacing NaN and whitespaces with missing string
dataset = pd.read_csv('data_cleaned.csv')
dataset = dataset.replace(to_replace={np.nan, ' '}, value='missing')

"""Now reading all dimensions' values into variables"""
# Get brands dimension
brands = dataset.iloc[0:199, 1].values

# Getting descriptions from JSON strings
descriptions_json = dataset.iloc[0:199, 2].values
# Via converting strings to objects
descriptions_objs = converter.getObjectFromJSONDescriptionsString(descriptions_json)
# Creating descriptions dimension and filling it up with descriptions acquired from the objects
descriptions = np.array([])
for i in range(len(descriptions_objs)):
    if descriptions_objs[i] != 'missing':
        descriptions = np.append(descriptions, descriptions_objs[i].value)
    else:
        descriptions = np.append(descriptions, 'missing')

# Getting features from JSON strings
features_json = dataset.iloc[0:199, 3].values
# Via converting strings to objects
features_objs = converter.getObjectFromJSONFeaturesString(features_json)
# Creating few dimensions from features objects
materials, genders, colors, sizes, models, categories = hf.getDimensionsFromFeatures(features_objs)

# Getting manufacturers dimension
manufacturers = dataset.iloc[0:199, 4].values

# Getting names dimension
names = dataset.iloc[0:199, 5].values

# Getting prices dimension and now treating not well formatted prices as missing
prices = dataset.iloc[0:199, 6].values
for i in range(len(prices)):
    if not hf.isnumeric(prices[i]):
        prices[i] = 'missing'

# Getting conditions dimension and keeping only the most common values
conditions = dataset.iloc[0:199, 7].values
for i in range(len(conditions)):
    if conditions[i] != 'new' and conditions[i] != 'New with tags' and \
            conditions[i] != 'New with box' and conditions[i] != 'New without box':
        conditions[i] = 'missing'

# Storing all data of each dimension in rawData
rawData = np.array([brands, descriptions, materials, genders, sizes,
                    models, categories, colors, manufacturers, names, prices])

# Transposing dimension values to make each column stand as a dimension and rows as data points
data = np.transpose(rawData)
# Creating pandas dataframe
dimensions = ['brand', 'description', 'material', 'gender', 'size', 'model',
              'category', 'color', 'manufacturer', 'name', 'price']
dataTable = pd.DataFrame(data, columns=dimensions)

"""Vectorizing data: doing One-hot encoding and Tfidf vectorization of 1,2,3-grams"""
# Creating arrays to store which dimensions to encode with different methods
# Dimensions to be one-hot encoded contained in this array
columnsToOneHot = ['brand', 'material', 'gender', 'size', 'model', 'category', 'color', 'manufacturer']
# Dimensions to tfidf vectorize their 1,2,3-grams contained in this array
columnsToTfidf = ['name', 'description']

# Using MyEncoder functions to
# get one-hot encoded data and their dimensions
dataOneHot, dimensionsOneHot = ec.oneHotEncodeDataFrameColumns(dataTable, columnsToOneHot)
# get tfidf vectors of 1,2,3-grams and their dimensions
dataTfidf, dimensionsTfidf = ec.tfdidfEncodeDataFrameColumns(dataTable, columnsToTfidf, 5000)
# Leftover data which is not transformed is price. Reshaping it to easily append all data together
priceData = dataTable['price'].values.reshape(-1, 1)
priceDimension = np.array([['price']])

# Appending all data together into transformedData
transformedData = np.append(dataOneHot, dataTfidf, axis=1)
transformedData = np.append(transformedData, priceData, axis=1)
# And dimensions to transformedDimensions
transformedDimensions = np.append(dimensionsOneHot, dimensionsTfidf)
transformedDimensions = np.append(transformedDimensions, priceDimension)

# Creating a new pandas DataFrame
newDataTable = pd.DataFrame(transformedData, columns=transformedDimensions)

"""At this point i saw 13 percent of data has missinf prices"""
missings = newDataTable['price'].values == 'missing'
#print(np.count_nonzero(missings) / np.size(missings))

# Removing these elements as they have no target variable
# Creating a new array cleanedData of same shape of transformedData to fill with items that all have price
cleanedData = np.empty((0, len(transformedData[0])))
for i in range(len(transformedData)):
    if transformedData[i][4117] != 'missing':
        cleanedData = np.append(cleanedData, transformedData[i].reshape(-1, transformedData.shape[1]), axis=0)

# This DataFrame contains all transformed data all with target variable. Might be ready to feed an algorithm with this
# data.
cleanedDataTable = pd.DataFrame(cleanedData, columns=transformedDimensions)

# Now 0 percent of data has missing items
missings = cleanedDataTable['price'].values == 'missing'
#print(np.count_nonzero(missings) / np.size(missings))
