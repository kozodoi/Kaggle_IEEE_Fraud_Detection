################################
#                              #
#       MEMORY REDUCTION       #
#                              #
################################

import numpy as np
def reduce_mem_usage(df, verbose = True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df





################################
#                              #
#    TEST-TIME AUGMENTATION    #
#                              #
################################

# creates multiple versionf of test data (with noise)
# averages predictions over the created samples

def predict_proba_with_tta(X_test, model, num_iteration, alpha = 0.01, n = 4, seed = 0):

    # set random seed
    np.random.seed(seed = seed)

    # original prediction
    preds = model.predict_proba(X_test, num_iteration = num_iteration)[:, 1] / (n + 1)

    # select numeric features
    num_vars = [var for var in X_test.columns if X_test[var].dtype != "object"]

    # synthetic predictions
    for i in range(n):

        # copy data
        X_new = X_test.copy()

        # introduce noise
        for var in num_vars:
            X_new[var] = X_new[var] + alpha * np.random.normal(0, 1, size = len(X_new)) * X_new[var].std()

        # predict probss
        preds_new = model.predict_proba(X_new, num_iteration = num_iteration)[:, 1]
        preds += preds_new / (n + 1)

    # return probs
    return preds


      
    
###############################
#                             #
#        ENCODE FACTORS       #
#                             #
###############################

# performs dummy or label encoding
import pandas as pd
def encode_factors(df, method = "label", skip = None):
    
    # select columns 
    factors = [f for f in df.columns if df[f].dtype == "object"]
    factors = [f for f in factors if f not in skip]
    
    # label encoding
    if method == "label":
        for var in factors:
            df[var], _ = pd.factorize(df[var])
        
    # dummy encoding
    if method == "dummy":
        df = pd.get_dummies(df, drop_first = True, columns = factors)
    
    # dataset
    return df


# performs label encoding
from sklearn.preprocessing import LabelEncoder
def label_encoding(df_train, df_valid, df_test):
    
    factors = [f for f in df_train.columns if df_train[f].dtype == 'object' or df_valid[f].dtype == 'object' or df_test[f].dtype == 'object']
    
    for f in factors:
        lbl = LabelEncoder()
        lbl.fit(list(df_train[f].values) + list(df_valid[f].values) + list(df_test[f].values))
        df_train[f] = lbl.transform(list(df_train[f].values))
        df_valid[f] = lbl.transform(list(df_valid[f].values))
        df_test[f]  = lbl.transform(list(df_test[f].values))

    return df_train, df_valid, df_test




###############################
#                             #
#        COUNT MISSINGS       #
#                             #
###############################

# computes missings per variable (count, %)
# displays variables with most missings

import pandas as pd
def count_missings(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending = False)
    table = pd.concat([total, percent], axis = 1, keys = ["Total", "Percent"])
    table = table[table["Total"] > 0]
    return table




###############################
#                             #
#        AGGRGEATE DATA       #
#                             #
###############################

# aggregates numeric data using specified stats
# aggregates factors using mode and nunique
# returns df with generated features

import scipy.stats
def aggregate_data(df, group_var, num_stats = ['mean', 'sum'], factors = None, var_label = None, sd_zeros = False):
    
    
    ### SEPARATE FEATURES
  
    # display info
    print("- Preparing the dataset...")

    # find factors
    if factors == None:
        df_factors = [f for f in df.columns if df[f].dtype == "object"]
    else:
        df_factors = factors
        df_factors.append(group_var)
        
    # partition subsets
    if type(group_var) == str:
        num_df = df[[group_var] + list(set(df.columns) - set(df_factors))]
        fac_df = df[df_factors]
    else:
        num_df = df[group_var + list(set(df.columns) - set(df_factors))]
        fac_df = df[df_factors] 
    
    # display info
    num_facs = fac_df.shape[1] - 1
    num_nums = num_df.shape[1] - 1
    print("- Extracted %.0f factors and %.0f numerics..." % (num_facs, num_nums))
    

    ##### AGGREGATION
 
    # aggregate numerics
    if (num_nums > 0):
        print("- Aggregating numeric features...")
        num_df = num_df.groupby([group_var]).agg(num_stats)
        num_df.columns = ["_".join(col).strip() for col in num_df.columns.values]
        num_df = num_df.sort_index()

    # aggregate factors
    if (num_facs > 0):
        print("- Aggregating factor features...")
        fac_int_df = fac_df.copy()
        for var in factors:
            fac_int_df[var], _ = pd.factorize(fac_int_df[var])
        fac_int_df[group_var] = fac_df[group_var]
        fac_df = fac_int_df.groupby([group_var]).agg([('count'), ('mode', lambda x: pd.Series.mode(x)[0])])
        fac_df.columns = ["_".join(col).strip() for col in fac_df.columns.values]
        fac_df = fac_df.sort_index()           


    ##### MERGER

    # merge numerics and factors
    if ((num_facs > 0) & (num_nums > 0)):
        agg_df = pd.concat([num_df, fac_df], axis = 1)
    
    # use factors only
    if ((num_facs > 0) & (num_nums == 0)):
        agg_df = fac_df
        
    # use numerics only
    if ((num_facs == 0) & (num_nums > 0)):
        agg_df = num_df
        

    ##### LAST STEPS

    # update labels
    if (var_label != None):
        agg_df.columns = [var_label + "_" + str(col) for col in agg_df.columns]
    
    # impute zeros for SD
    if (sd_zeros == True):
        stdevs = agg_df.filter(like = "_std").columns
        for var in stdevs:
            agg_df[var].fillna(0, inplace = True)
            
    # dataset
    agg_df = agg_df.reset_index()
    print("- Final dimensions:", agg_df.shape)
    return agg_df




###############################
#                             #
#      ADD DATE FEATURES      #
#                             #
###############################

# creates a set of date-related features
# outputs df with generated features

def add_datepart(df, fldname, drop = True, time = False):
    "Helper function that adds columns relevant to a date."
    
    fld = df[fldname]
    fld_dtype = fld.dtype
    
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
        
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    
    if time: attr = attr + ['Hour', 'Minute', 'Second']
        
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    
    if drop: df.drop(fldname, axis=1, inplace=True)
      
    


###############################
#                             #
#      ADD TEXT FEATURES      #
#                             #
###############################

# extract basic features from strings
# appends new features to the data frame

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
def add_text_features(data, strings, k = 5, keep = True):

    ##### PROCESSING LOOP
    for var in strings:

        ### TEXT PREPROCESSING

        # replace NaN with empty string
        data[var][pd.isnull(data[var])] = ''

        # remove common words
        freq = pd.Series(' '.join(data[var]).split()).value_counts()[:10]
        #freq = list(freq.index)
        #data[var] = data[var].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
        #data[var].head()

        # remove rare words
        freq = pd.Series(' '.join(data[var]).split()).value_counts()[-10:]
        #freq = list(freq.index)
        #data[var] = data[var].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
        #data[var].head()

        # convert to lowercase 
        data[var] = data[var].apply(lambda x: " ".join(x.lower() for x in x.split())) 

        # remove punctuation
        data[var] = data[var].str.replace('[^\w\s]','')         


        ### COMPUTE BASIC FEATURES

        # word count
        data[var + '_word_count'] = data[var].apply(lambda x: len(str(x).split(" ")))
        data[var + '_word_count'][data[var] == ''] = 0

        # character count
        data[var + '_char_count'] = data[var].str.len().fillna(0).astype('int64')


        ##### COMPUTE TF-IDF FEATURES

        # import vectorizer
        tfidf  = TfidfVectorizer(max_features = k, 
                                 lowercase    = True, 
                                 norm         = 'l2', 
                                 analyzer     = 'word', 
                                 stop_words   = 'english', 
                                 ngram_range  = (1, 1))

        # compute TF-IDF
        vals = tfidf.fit_transform(data[var])
        vals = pd.SparseDataFrame(vals)
        vals.columns = [var + '_tfidf_' + str(p) for p in vals.columns]
        data = pd.concat([data, vals], axis = 1)


        ### CORRECTIONS

        # remove raw text
        if keep == False:
            del data[var]

        # print dimensions
        #print(data.shape)
        
    # return data
    return data