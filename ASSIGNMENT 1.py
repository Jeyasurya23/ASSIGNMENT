from google.colab import drive

drive.mount('/content/drive')
#Mounted at /content/drive

def import_housing_data(url):
     import pandas as pd
     df = pd.read_csv(url)
     df.drop(columns=['Id'])
     return df

df =import_housing_data('http://www.ishelp.info/data/housing_full.csv')
df.head()

def unistats(df):
 import pandas as pd
 output_df =pd.DataFrame(columns=['Count','Missing','Unique','Dtype','Mean','Mode','Min','25%','Median','75%','Max','Std','Skew','Kurt'])
 for col in df:
     if pd.api.types.is_numeric_dtype(df[col]): 
     output_df.loc[col] =
    [df[col].count(),df[col].isnull().sum(),df[col].nunique,df[col].nuniqu
    e(),df[col].dtype,pd.api.types.is_numeric_dtype(df[col]),df[col].mode(
    ).values[0],df[col].mean(),df[col].mean(),df[col].min(),df[col].quantile(0.25),df[col].median(),df[col].quantile(0.75),df[col].max(),df[col].std(),df[col].skew(),df[col].kurt()]
     else:
     output_df.loc[col] =
    [df[col].count(),df[col].isnull().sum(),df[col].nunique,df[col].nuniqu
    e(),df[col].dtype,pd.api.types.is_numeric_dtype(df[col]),df[col].mode(
    ).values[0],
     '-','-','-','-''-','-','-','-','-']
 return output_df.sort_values(by=['Numeric','Unique'],ascending=False)
#Test the Function

import pandas as pf
pandas.set_option('display.max_rows',100)
pandas.set_option('display.max_columns',100)
df = pd.read_csv('http://www.ishelp.info/data/housing_full.csv')
unistats(df)

def anova(df, feature, label):
     import pandas as pd
     import numpy as np
     from scipy import stats
     groups = df[feature].unique()
     df_grouped = df.groupby(feature)
     group_labels = []
     for g in groups:
         g_list = df_grouped.get_group(g)
         group_labels.append(g_list[label])
 return stats.f_oneway(*group_labels)

# Bivariate: Numeric to numeric: Correlation
# Bivariate: Numeric to categorical: one-way ANOVA (3+ groups) or ttest (2 groups)
# Bivariate: categorical to categorical: Chi-square

def bivstats(df, label):
     from scipy import stats
     import pandas as pd
     import numpy as np
     # Create an empty DataFrame to store output
     output_df = pd.DataFrame(columns=['stat', '+/-', 'Effect size', 'pvalue'])
     for col in df:
         if not col == label:
             if df[col].isnull().sum() == 0:
                 if pd.api.types.is_numeric_dtype(df[col]):
                 r, p = stats.pearsonr(df[label], df[col])
                 output_df.loc[col] = ['r', np.sign(r), abs(round(r, 3)), 
                round(p, 6)]
                 else:
                 F, p = anova(df[[col, label]], col, label)
                 output_df.loc[col] = ['F', '', round(F, 3), round(p, 6)]
         else:
         output_df.loc[col] = [np.nan, np.nan, np.nan, 'nulls']
 return output_df.sort_values(by=['Effect size', 'stat'], 

ascending=[False, False])
import pandas as pd
pd.options.display.float_format = '{:.5f}'.format
df = pd.read_csv('http://www.ishelp.info/data/housing_full.csv')
bivstats(df, 'SalePrice')

def import_housing_data(url):
     df = pd.read_csv(url)
     df.drop(columns=['Id'], inplace=True)
     df.dropna(axis=1, inplace=True)
     for col in df:
         if col[0].isdigit():
             nums = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 
            'seven', 'eight', 'nine']
             df. rename(columns={col:nums[int(col[0])] + '_' + col}, 
            inplace=True)
 return df

import sys
sys.path.append('/content/drive/My Drive/ColabNotebooks/')
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.options.display.float_format = '{:.8f}'.format
df = import_housing_data('http://ishelp.info/data/housing_full.csv')
df.head()

def import_housing_data(df, label):
     import numpy as np
     import pandas as pd
     import statsmodels.api as sm
     from sklearn import preprocessing
     label = 'SalePrice'
     for col in df:
         if not pd.api.types.is_numeric_dtype(df[col]):
         df = df.join(pd.get_dummies(df[col], prefix=col, 
        drop_first=False))
         df = df.select_dtypes(np.number)

    d_minmax =pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df), 
    columns=df.columns)
     y = df_minmax[label] 
     x = df_minmax.drop(columns=[label, 'Utilities AllPub', 
    'Exteriorist_BrkComm']).assign(const=1)
     results = sm.OLS(y, x).fit()
     results.summary()
