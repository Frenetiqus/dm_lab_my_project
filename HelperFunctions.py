import numpy as np

# This function checks if input is numeric
def isnumeric(s):
    return all(c in "0123456789.+-" for c in s) and any(c in "0123456789" for c in s)

# This function's input is an array containing feature objects that are previously acquired from JSON strings. It
# returns arrays containing values from the specified dimensions for each data point.
def getDimensionsFromFeatures(features_objs):
    materials = np.array([])
    genders = np.array([])
    sizes = np.array([])
    models = np.array([])
    categories = np.array([])
    colors = np.array([])
    for x in features_objs:
        material, gender, color, size, model, category = np.array(['missing']), np.array(['missing']), \
                                                         np.array(['missing']), np.array(['missing']), \
                                                         np.array(['missing']), np.array(['missing'])
        if x != 'missing':
            for x_i in x:
                if x_i.key == 'Material':
                    material = np.array([x_i.value[0]])
                elif x_i.key == 'Gender':
                    gender = np.array([x_i.value[0]])
                elif x_i.key == 'Color':
                    color = np.array([x_i.value[0]])
                elif x_i.key == 'Shoe Size' or x_i.key == 'Size':
                    size = np.array([x_i.value[0]])
                elif x_i.key == 'Model':
                    model = np.array([x_i.value[0]])
                elif x_i.key == 'Shoe Category' or x_i.key == 'Category':
                    category = np.array([x_i.value])
        materials = np.append(materials, material)
        genders = np.append(genders, gender)
        colors = np.append(colors, color)
        sizes = np.append(sizes, size)
        models = np.append(models, model)
        categories = np.append(categories, category)
    return materials, genders, colors, sizes, models, categories
