# dm_lab_my_project
 This repo contains data which I worked with and also python code in which I also commented most lines and functions. My work mostly consists of getting values from file and transforming them into feedable form.

I spent most of my time extracting data from files as some columns contained JSON objects as strings so I had to find a way to properly extract these values. 
I extracted them and read these values into variables representing dimensions. I continued on transforming these dimensions' values into a form that can represent most of the information contained in them.
I used one-hot encoding technique for categorical variables such as condition, brand, size or color. For values in dimensions such as name and description I used Tfidf vectorization of 1,2,3-grams as these values mostly consist of many characters, words. This vectorization technique is good solution to keep track of information of each individual string based on all strings and also these vectors contain 2 and 3 grams of all strings so this ensures that the locational information of words is preserved as well. Also after strings are n-grammed a tfdidf transformation is applied so each gram in a string is weighted based on the transformation which weighs more unique words and less words that are common like 'a', 'the'.

Basically I extracted data from a csv file then transformed and saved it in a variable and also in a pandas DataFrame.

Files:

 main.py : contains all code for extracting and transforming
 
 JSONStringConverter.py : contains functions to extract data from JSON strings
 
 HelperFunctions.py : some self-made functions to help keep a cleaner code
 
 MyEncoders.py : contains functions that do the transformations on data
 
 data_200.csv : original data containing only 200 data points
 
 data_cleaned.csv : the former csv filed cleaned from unneccessary columns
