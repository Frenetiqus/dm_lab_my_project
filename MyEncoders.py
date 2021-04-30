from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

# This function uses one-hot encoding method on an array of values creating same number of elements with more
# dimensions.
def oneHotEncode(values):
    vectorizer = CountVectorizer()
    oneHot = vectorizer.fit_transform(values)
    dataOneHot = oneHot.toarray()
    dimensionsOneHot: list = vectorizer.get_feature_names()
    return dataOneHot, dimensionsOneHot

# This function uses tfdidf vectorization of 1,2 and 3-grams of elements contained in values array. Max features can
# be specified as max_features argument.
def tfidfEncode(values, max_features):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=max_features)
    tfidf = vectorizer.fit_transform(values)
    dataTfidf = tfidf.toarray()
    dimensionsTfidf = vectorizer.get_feature_names()
    return dataTfidf, dimensionsTfidf


# This function uses one-hot encoding method on multiple arrays from a pandas DataFrame specified by an
# array: columnsToOneHot containing which columns to encode from the DataFrame.
def oneHotEncodeDataFrameColumns(dataTable, columnsToOneHot):
    columnDataOneHot, columnDimensionsOneHot = oneHotEncode(dataTable[columnsToOneHot[0]].values)
    dataOneHot = columnDataOneHot
    dataDimenensionOneHot = columnDimensionsOneHot
    for i in range(1, len(columnsToOneHot)):
        columnDataOneHot, columnDimensionsOneHot = oneHotEncode(dataTable[columnsToOneHot[i]].values)
        dataOneHot = np.append(dataOneHot, columnDataOneHot, axis=1)
        dataDimenensionOneHot = np.append(dataDimenensionOneHot, columnDimensionsOneHot)
    return dataOneHot, dataDimenensionOneHot


# This function uses tfdidf vectorization of 1,2 and 3-grams on elements of multiple arrays acquired from a pandas
# DataFrame specified by an array: columnsToTfidf containing which columns to encode from the DataFrame.
# Max features can be specified as max_features argument.
def tfdidfEncodeDataFrameColumns(dataTable, columnsToTfidf, max_features):
    columnDataTfidf, columnDimensionsTfidf = tfidfEncode(dataTable[columnsToTfidf[0]].values, max_features)
    dataTfidf = columnDataTfidf
    dataDimenensionTfidf = columnDimensionsTfidf
    for i in range(1, len(columnsToTfidf)):
        columnDataTfidf, columnDimensionsTfidf = tfidfEncode(dataTable[columnsToTfidf[i]].values, max_features)
        dataTfidf = np.append(dataTfidf, columnDataTfidf, axis=1)
        dataDimenensionTfidf = np.append(dataDimenensionTfidf, columnDimensionsTfidf)
    return dataTfidf, dataDimenensionTfidf
