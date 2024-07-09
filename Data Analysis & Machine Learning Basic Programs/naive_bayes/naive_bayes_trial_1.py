import pandas as pd
import numpy as np

data = pd.DataFrame()

data['Height'] = [6,5.92,5.58,5.92,5.5,5.42,5.58,5]
data['Weight'] = [180,190,170,165,100,150,130,150]
data['FootSize'] = [12,11,12,10,6,8,7,9]

data['Gender']=['male','male','male','male','female','female','female','female',]
print(data)

person = pd.DataFrame()
person['Height'] = [5]
person['Weight'] = [130]
person['FootSize'] = [8]

n_male=data['Gender'][data['Gender']=='male'].count()
n_female=data['Gender'][data['Gender']=='female'].count()
total_ppl=data['Gender'].count()

data_mean=data.groupby('Gender').mean()
print(data_mean)

data_variance=data.groupby('Gender').var()
print(data_variance)

male_height_mean = data_mean['Height'][data_variance.index=='male'].values[0]
male_weight_mean = data_mean['Weight'][data_variance.index=='male'].values[0]
male_footsize_mean = data_mean['FootSize'][data_variance.index=='male'].values[0]

male_height_var = data_variance['Height'][data_variance.index=='male'].values[0]
male_weight_var = data_variance['Weight'][data_variance.index=='male'].values[0]
male_footsize_var = data_variance['FootSize'][data_variance.index=='male'].values[0]


female_height_mean = data_mean['Height'][data_variance.index=='female'].values[0]
female_weight_mean = data_mean['Weight'][data_variance.index=='female'].values[0]
female_footsize_mean = data_mean['FootSize'][data_variance.index=='female'].values[0]

female_height_var = data_variance['Height'][data_variance.index=='female'].values[0]
female_weight_var = data_variance['Weight'][data_variance.index=='female'].values[0]
female_footsize_var = data_variance['FootSize'][data_variance.index=='female'].values[0]

def p_xgiven_y(x,mean_y,Varience_y):
    p=1/(np.sqrt(2*np.pi*Varience_y))*np.exp((-(x-mean_y)**2)/(2*Varience_y))
    print(p)
    
p_xgiven_y(person['Height'][0],male_height_mean , male_height_var)    
p_xgiven_y(person['Weight'][0],male_weight_mean , male_weight_var)    
p_xgiven_y(person['FootSize'][0],male_footsize_mean , male_footsize_var)    
    
    
p_xgiven_y(person['Height'][0],female_height_mean , female_height_var)    
p_xgiven_y(person['Weight'][0],female_weight_mean , female_weight_var)    
p_xgiven_y(person['FootSize'][0],female_footsize_mean , female_footsize_var)

    
            
    
        
    