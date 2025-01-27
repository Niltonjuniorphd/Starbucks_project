# Starbucks_project

Apply Data Science and Machine Learning in Python to analyze Starbucks customers' behavior and predict which customers will respond to an offer.

# Project Description
There are EDA, clustering and machine learning techniques applyed to analyze a dataset containing simulated data about customers of a coffee shop chain.

# Data understanding: three json files:

**portfolio.json:** Type and content of each offer.
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

```bash
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 6 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   reward      10 non-null     int64 
 1   channels    10 non-null     object
 2   difficulty  10 non-null     int64 
 3   duration    10 non-null     int64 
 4   offer_type  10 non-null     object
 5   id          10 non-null     object
dtypes: int64(3), object(3)
memory usage: 612.0+ bytes
```

-> it's cleaned and ready to be used

**profile.json**: Customer demographic data for each customer.
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

```bash	
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 17000 entries, 0 to 16999
Data columns (total 5 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   gender            14825 non-null  object 
 1   age               17000 non-null  int64  
 2   id                17000 non-null  object 
 3   became_member_on  17000 non-null  int64  
 4   income            14825 non-null  float64
dtypes: float64(1), int64(2), object(2)
memory usage: 664.2+ K
```
-> the `'gender'` and `'income'` columns have NaN values.

**transcript.json**: Event log including, event type by offer ids and transaction amounts (transaction is not correlated to the offer id).
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

```bash
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 306534 entries, 0 to 306533
Data columns (total 4 columns):
 #   Column  Non-Null Count   Dtype 
---  ------  --------------   ----- 
 0   person  306534 non-null  object
 1   event   306534 non-null  object
 2   value   306534 non-null  object
 3   time    306534 non-null  int64 
dtypes: int64(1), object(3)
memory usage: 9.4+ MB
```

-> the `event` column has 4 different values: `offer received`, `offer viewed`, `offer completed` and `transaction`.
```bash
transcript_raw['event'].value_counts()

# output:
event
transaction        138953
offer received      76277
offer viewed        57725
offer completed     33579
Name: count, dtype: int64
```

-> the `value` column has dictioneries with 3 different keys: `offer id`, `offer_id` and `amount`.
```bash
{[*x][0] for x in transcript_raw['value']}

# output:
{'amount', 'offer id', 'offer_id'}
```

# Data Cleaning and Formatting Plan

**Objective**:
- Standardize columns and variables names for all datasets. 
- Filter the `transaction` events from the `transcript` dataset.
- Crate a unique dataset (transcript_collections dataframe ) mergin the three datasets.
- Handling missing values appropriately.
- Check for if existis loosing data problems.
- Check for duplicates.
---

- **Profile Dataset:**  Convert the `became_member_on` column to a standardized **datetime** format for consistency and easier analysis. 

```bash
profile = profile_raw.copy(deep=True)

# Convert the 'became_member_on' column to a datetime format
profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d')

# Create a new column with only the year and month of the membership
profile['bec_memb_year_month'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d').dt.strftime('%Y-%m')
```

---

- **Portfolio Dataset:** Changing the `id` column values to a more readeble tag. 

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXUpdate column names to make them more descriptive and easier to read.

```bash	
# noting to do here
portfolio = portfolio_raw.copy()
```
---


- **Transcript Dataset:** **Step-1:** Insert `'_'` in the values of the `'event'` column and modify the dictionary key in the `'value'` column by replacing spaces (`' '`) with underscores (`'_'`). **Step-2:** Normalize the `'value'` column and create a new dataframe (transcript_b).

```bash
# Standardisation of column names
transcript = transcript_raw.copy(deep=True)
transcript['event'] = transcript['event'].str.replace(' ','_')

# Fixing the dicts key names
def fix_offer_id(value):
    if isinstance(value, dict) and 'offer id' in value:
        value['offer_id'] = value.pop('offer id')
    return value

transcript['value'] = transcript['value'].apply(fix_offer_id)

# Normalizing the value column and create a new dataframe (transcript_b)
value_df = pd.json_normalize(transcript['value'])
transcript_b = pd.concat([transcript, value_df], axis=1).drop('value', axis=1)

transcript_b.info()

# output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 306534 entries, 0 to 306533
Data columns (total 6 columns):
 #   Column    Non-Null Count   Dtype  
---  ------    --------------   -----  
 0   person    306534 non-null  object 
 1   event     306534 non-null  object 
 2   time      306534 non-null  int64  
 3   offer_id  167581 non-null  object 
 4   amount    138953 non-null  float64
 5   reward    33579 non-null   float64
dtypes: float64(2), int64(1), object(3)
memory usage: 14.0+ MB
```
---
# Filtering the `'transaction'` events from the `transcript_b` dataset.

Since the `transaction` events don't have a reference to any `offer_id`, the transaction events can be separated from the `transcript_b` dataset. (However, there is a relationship where the time value of the `person`-`offer_completed`-`time` event and the `person`-`transaction`-`time` event are the same and this will be explored).

```bash
transactions = transcript_b.loc[transcript_b['event'] == 'transaction',:]
transcript_c = transcript_b.loc[transcript_b['event'] != 'transaction',:]
transcript_c.info()
```

---
# Merging the three datasets into a single dataset:

**objective**: Create a single dataset that contains all the information from the three datasets.
```bash
transcript_collection = transcript_c.merge(profile, left_on='person', right_on='id', how='left').drop('id', axis=1)
transcript_collection = transcript_collection.merge(portfolio, left_on='offer_id', right_on='id', how='left', suffixes=(' ', '_std')).drop('id', axis=1)

transcript_collection.info()

# output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 167581 entries, 0 to 167580
Data columns (total 16 columns):
 #   Column               Non-Null Count   Dtype         
---  ------               --------------   -----         
 0   person               167581 non-null  object        
 1   event                167581 non-null  object        
 2   time                 167581 non-null  int64         
 3   offer_id             167581 non-null  object        
 4   amount               0 non-null       float64       
 5   reward               33579 non-null   float64       
 6   gender               148805 non-null  object        
 7   age                  167581 non-null  int64         
 8   became_member_on     167581 non-null  datetime64[ns]
 9   income               148805 non-null  float64       
 10  bec_memb_year_month  167581 non-null  object        
 11  reward_std           167581 non-null  int64         
 12  channels             167581 non-null  object        
 13  difficulty           167581 non-null  int64         
 14  duration             167581 non-null  int64         
 15  offer_type           167581 non-null  object        
dtypes: datetime64[ns](1), float64(3), int64(5), object(7)
memory usage: 20.5+ MB
```

- Modify the `offer_id` column so that it contains a more readable tag.
```bash
port_id = {
    'ae264e3637204a6fb9bb56bc8210ddfd': 'ofr_A',
    '4d5c57ea9a6940dd891ad53e9dbe8da0': 'ofr_B',
    '3f207df678b143eea3cee63160fa8bed': 'ofr_C',
    '9b98b8c7a33c4b65b9aebfe6a799e6d9': 'ofr_D',
    '0b1e1539f2cc45b7b9fa7c272da2e1d7': 'ofr_E',
    '2298d6c36e964ae4a3e7e9706d1fb8c2': 'ofr_F',
    'fafdcd668e3743c1bb461111dcafc2a4': 'ofr_G',
    '5a8bc65990b245e5a138643cd4eb9837': 'ofr_H',
    'f19421c1d4aa40978ebb69ca19b0e20d': 'ofr_I',
    '2906b810c7d4411798c6938adc9daaa5': 'ofr_J'
}

transcript_collection['ofr_id_short'] = transcript_collection['offer_id'].map(port_id)
transcript_collection = transcript_collection.drop(['offer_id'], axis=1)

# create a new column that contains the number of channels
transcript_collection['channels_count'] = transcript_collection['channels'].apply(lambda x: len(x))


```

---

# Handling Missing Values

## **`transactions` Dataset:**

```bash
<class 'pandas.core.frame.DataFrame'>
Index: 138953 entries, 12654 to 306533
Data columns (total 6 columns):
 #   Column    Non-Null Count   Dtype  
---  ------    --------------   -----  
 0   person    138953 non-null  object 
 1   event     138953 non-null  object 
 2   time      138953 non-null  int64  
 3   offer_id  0 non-null       object 
 4   amount    138953 non-null  float64
 5   reward    0 non-null       float64
dtypes: float64(2), int64(1), object(3)
memory usage: 7.4+ MB
```
This dataset contains all the transactions during the period of the data collection. Just drop the column `offer_id` and `reward`.

```bash
transactions = transactions.dropna(axis=1)
```
---

## **`transcript_collections` Dataset:**

```bash
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 167581 entries, 0 to 167580
Data columns (total 16 columns):
 #   Column               Non-Null Count   Dtype         
---  ------               --------------   -----         
 0   person               167581 non-null  object        
 1   event                167581 non-null  object        
 2   time                 167581 non-null  int64         
 3   amount               0 non-null       float64       
 4   reward               33579 non-null   float64       
 5   gender               148805 non-null  object        
 6   age                  167581 non-null  int64         
 7   became_member_on     167581 non-null  datetime64[ns]
 8   income               148805 non-null  float64       
 9   bec_memb_year_month  167581 non-null  object        
 10  reward_std           167581 non-null  int64         
 11  channels             167581 non-null  object        
 12  difficulty           167581 non-null  int64         
 13  duration             167581 non-null  int64         
 14  offer_type           167581 non-null  object        
 15  ofr_id_short         167581 non-null  object        
dtypes: datetime64[ns](1), float64(3), int64(5), object(7)
memory usage: 20.5+ MB
```

- Drop the column `amount`
- The column `reward` has (134002) missing values because the `event` associated to it is only the `offer_completed`.
- The column `gender` has (18776) missing values and and three categories: `F`, `M`, `O`.
- The column `income` has (18776) missing values, the same as the column `gender`. (Two strategy will be adopted: 1- drop the data and 2- modeling and predict the missing values)
