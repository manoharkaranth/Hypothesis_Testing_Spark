#!/usr/bin/env python
# coding: utf-8

# <font size='4'><b>Problem statement:</b> In the television documentary “Ross Kemp and the Armed Police” broadcast 6th September 2018 by ITV, multiple claims were made regarding violent crime in the UK.<br/><br/>
# These claims were:
# 1.	Violent Crime is increasing
# 2.	There are more firearms incidents per head in Birmingham than anywhere else in the UK
# 3.	Crimes involving firearms are closely associated with drugs offences<br/><br/>
# In this assignment you will investigate these claims using real, publicly available data sets that will be made available to you and placed in Amazon S3. </font>

# <br/><br/>

# In[ ]:


# Importing count function
from pyspark.sql.functions import count,col
import pyspark.sql.functions as func

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Ignoring warnings.
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Use this path for big data.
# import urllib.request
# urllib.request.urlretrieve("https://s3.amazonaws.com/kf7032-20.northumbria.ac.uk/all_crimes18_hdr.txt.gz")


# In[ ]:


# Moving data from 'temp' folder to Databricks file system
# dbutils.fs.mv("file:/tmp/tmp_jaq8rci","dbfs:/data/all_crimes18_hdr.txt.gz")
# Output: True indicates successful move.


# In[ ]:


# Reading the text file
txt=sc.textFile("FileStore/tables/crimes_2018_only_txt.gz")
# Assigning the first line to header.


# ##### Note: 2018 dataset does not have header, we need to set it manually.

# In[ ]:


header='Crime_ID,Month,Reported_by,Falls_within,Longitude,Latitude,Location,LSOA_code,LSOA_name,Crime_type,Last_outcome_category,Context'


# In[ ]:


txt.take(10)


# ##### Note: Below step is very important as data contained ',' values embedded in strings which was causing numerous errors in Schema. Here, we are rejecting all the lines which have ',' value embedded in string. i.e. we are allowing only 12 values. If after spliting, number of values is more than 12, it clearly indicats occurance of ',' within strings.

# In[ ]:


# Filtering corrupt data.
txt = txt.filter(lambda line: len(line.split(","))==12)


# In[ ]:


# Assigning split values to a temporary variable.
temp_var = txt.map(lambda k: k.split(","))


# In[ ]:


# Converting data in the temporary variable into a Spark dataframe.
df=temp_var.toDF(header.split(","))


# In[ ]:


# Taking a peek at the dataframe.
df.show(10)


# In[ ]:


# Droping columns which have no value as a part of data cleaning.
df_new=df.drop('Crime_ID','Last_outcome_category','Context')


# In[ ]:


# Observing the schema of dataframe.
df_new.printSchema()


# ##### Note: Here, all variables are strings. But we know that 'Month' should take date type and Longitude & Longitudes should take double type

# In[ ]:


# Converting these three variables into required types.
df_new=df_new.withColumn("Month",df['Month'].cast("date"))
df_new=df_new.withColumn("Longitude",df['Longitude'].cast("double"))
df_new=df_new.withColumn("Latitude",df['Latitude'].cast("double"))


# In[ ]:


# Checking the schema again.
df_new.printSchema()
# Awesome, we are now ready with formatted data.


# ##### Let's see the descriptive statistics

# In[ ]:


# Statistical analysis of variables.
df_new.describe().show()


# ##### Claim-1: Violent crimes are increasing in UK.

# In[ ]:


# Let's see the number of crime types in UK.
df_new.groupby('Crime_type').count().show()


# In[ ]:


# Creating a temporary table for SQL queries. Table name is Crime_UK.
temp_table_name = "Crime_UK"
df_new.createOrReplaceTempView(temp_table_name)


# ##### Note: The above table is temporary and availabe only during this run time

# In[ ]:


# Just see how the table is
get_ipython().run_line_magic('sql', '')
select * from Crime_UK limit 3;


# In[ ]:


# This is an important funtion used throughtout the code for count aggregation.
import pyspark.sql.functions as F

cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))


# In[ ]:


# Analyzing both types of violent crimes: 'Violent crime' and 'Violence and sexual offences'.
get_ipython().run_line_magic('sql', '')
select * from Crime_UK where Crime_type=='Violent crime'or Crime_type=='Violence and sexual offences' order by Month limit 3;


# ##### Note: Here I am considering 'Violent Crime' and 'Violence and sexual offences' as violent crimes for the problem statement. I am not considering robbery

# In[ ]:


# Dataframe for violent crime.
df_time_series=df_new.groupBy('Month').agg(cnt_cond(F.col('Crime_type').isin(['Violent crime', 'Violence and sexual offences'])).alias('count'))
df_time_series.show(3)


# In[ ]:


# Converting spark dataframe into pandas dataframe to leverage matplotlib.
pdf = (
    df_time_series.select(
        "Month",
        "count")
    .orderBy("Month")
    .toPandas()
)


# In[ ]:


# Calculating rolling average by keeping window as 2.
pdf['rolling average'] = pdf['count'].rolling(2).mean().shift(-1)
pdf['rolling average'].head(3)


# In[ ]:


# ploting a time series plot
plt.figure(figsize=(9,6))
sns.lineplot(x="Month",y="count",
             data=pdf,
             ci=None)
sns.lineplot(x="Month",y="rolling average",
             data=pdf,
             ci=None)

plt.xlabel("Date", size=14)
plt.ylabel("number of violent crimes", size=14)


# ##### Validation of claim-1: Violent crime rate took a dip in February 2018. However, it started to increase afterwards except a little decrease in April and June. It can be clearly proved that violent crimes are increasing in UK. February decrease is possibly due to reduction in the temperature. Increase in crime rate with increasing temperature is a proven phenomenon in cold countries across the globe.

# <br/><br/>

# ##### Claim-2: There are more firearm incidents per head in Birmingham than anywhere else in UK

# In[ ]:


# Loading LSOA dataset which will be merged with Crime data later.
df1 = spark.read.format("csv")  .option("inferSchema", "true")   .option("header", "true")   .option("sep", ',')  .load("dbfs:/FileStore/shared_uploads/X@gmail.com/LSOA_pop_v2.csv") 
df1.show(3)


# In[ ]:


print(df1.columns)


# In[ ]:


df1= df1.withColumnRenamed('Variable: All usual residents; measures: Value', 'Number_of_People')
df1= df1.withColumnRenamed('Variable: Males; measures: Value', 'Number_of_Males')
df1= df1.withColumnRenamed('Variable: Females; measures: Value','Number_of_Females')
df1.show(3)


# In[ ]:


# Slicing the dataframe keeping only required data.
df1_sel=df1.select(['geography code','Number_of_People'])
df1_sel.show(3)


# In[ ]:


df1_sel.printSchema()


# In[ ]:


df1_sel= df1_sel.withColumnRenamed('geography code', 'LSOA_code')


# In[ ]:


# Slicing the crime dataframe
df_sel=df_new.select(['Month','LSOA_code', 'LSOA_name','Crime_type'])
df_sel.show(3)


# In[ ]:


df_sel = df_sel.withColumn("LSOA_6", df_sel.LSOA_name.substr(1,6))
df_sel.show(3)


# In[ ]:


# Merging crime and LSOA slices
df_joined= df_sel.join(df1_sel, on='LSOA_code', how='left_outer')
df_joined.show(3)


# In[ ]:


# Df for firearm incidents.
df_firearm=df_joined.groupBy('LSOA_6').agg(
    cnt_cond(F.col('Crime_type').isin(['Possession of weapons', 'Public disorder and weapons'])).alias('y_cnt'))
df_firearm.show(3)


# In[ ]:


# Grouping the population by LSOA
df_population = df_joined.groupby("LSOA_6").    agg(
        func.sumDistinct("Number_of_People").alias("total_population")
    )


# In[ ]:


# look at the population data and cross check on internet
df_population.show(3)
# Population data is validated through internet information


# In[ ]:


# Joining population and firearms dataframes.
df_per_capita= df_firearm.join(df_population, on='LSOA_6', how='left_outer')
df_per_capita.show(3)


# In[ ]:


# Calculating firearm incidents per head
df_per_capita=df_per_capita.withColumn('per_head', df_per_capita['y_cnt']/df_per_capita['total_population'])


# In[ ]:


df_per_capita.show(3)


# In[ ]:


# Converting into pandas dataframe
pdf1 = (
    df_per_capita.select(
        "LSOA_6",
        "per_head")
    .orderBy("per_head")
    .toPandas()
)


# In[ ]:


# Looking at top 30 highest firearm/ person regions
pdf1.tail(30)


# In[ ]:


# Let's select only top 30 out of 309 LSOA for easy dipiction
pdf_sel=pdf1.iloc[-30:,:]
pdf_sel.head(3)


# In[ ]:


pdf_sel["per_ten_thousand"]=pdf_sel['per_head']*10000
pdf_sel.head(3)


# In[ ]:


plt.bar(pdf_sel.LSOA_6,pdf_sel.per_ten_thousand)
plt.xticks(rotation=90)
plt.title(" Firearm per 10,000 people")
plt.xlabel("Top 30 LSOAs")
plt.ylabel('Incidents per 10,000 people')


# ##### Validation of claim-2: By looking at the bar plot, Birmingham holds 29th position in UK out of 309 LSOAs in terms of fire arm incidents per head. So, we can clearly reject the claim made in ITV documentary. Uttlesford is holding the first position with 64 incidents per 10,000 people. City of London which is arguably the most metropolitan city in UK seems to be following US culture being second only to Uttlesford.¶

# <br/><br/>

# #### Claim 3: Crimes involving firearms are closely associated with drugs offences

# In[ ]:


# A dataframe for LSOA wise drug incidents.
df_drugs=df_joined.groupBy('LSOA_6').agg(cnt_cond(F.col('Crime_type') =='Drugs').alias('drugs_count'))
df_drugs.show(3)


# In[ ]:


# Joining firearm and drug dataset for further analysis
df_final= df_firearm.join(df_drugs, on='LSOA_6', how='left_outer')
df_final.show(3)


# In[ ]:


# Renaming the firearm count column
df_final= df_final.withColumnRenamed('y_cnt','firearms_count')
df_final.show(3)


# In[ ]:


# Let's check Pearson correlation first
df_final.stat.corr('firearms_count','drugs_count')


# ##### Note: Spearman correlation is not supported with Spark df. So, converting into Pandas df to do so.

# In[ ]:


# Converting into pandas dataframe
pdf2 = (
    df_final.select(
        "firearms_count",
        "drugs_count")
    .toPandas()
)


# In[ ]:


# Spearman correlation.
pdf2.corr(method='spearman', min_periods=1)


# In[ ]:


# There was an outlier giving trouble in viz. Removing that outlier here.
pdf2_sel=pdf2[pdf2['firearms_count']<1000]


# In[ ]:


# A scatter plot to visualise the relationship.
plt.scatter(pdf2_sel.firearms_count,pdf2_sel.drugs_count)
plt.title(" Firearms vs Drugs")
plt.xlabel("Firearms")
plt.ylabel('Drugs')


# In[ ]:


plt.close()


# In[ ]:


# Plotting histograms for comparison
fig, ax = plt.subplots(1,2)
ax[0].hist(pdf2_sel['drugs_count'],alpha=0.7, bins=35, color='green')
ax[0].title.set_text('drugs')
ax[0].set_xlabel('count of drug incidents')
ax[0].set_ylabel('number of occurances')

ax[1].title.set_text('firearms count')
ax[1].hist(pdf2_sel['firearms_count'], alpha=0.8, bins=35, color='red')
ax[1].set_xlabel('count of firearm incidents')
ax[1].set_ylabel('number of occurances')


plt.show()


# In[ ]:


# Let's do clustering and see how it pans out. 
vec = VectorAssembler(inputCols=["firearms_count", "drugs_count"], outputCol="features")
new_df = vec.transform(df_final)
new_df.show()


# In[ ]:


# Applying clustering to find more insights.
kmeans = KMeans(k=4, seed=1) 
model = kmeans.fit(new_df.select('features'))


# In[ ]:


transformed = model.transform(new_df)
transformed.show(3)  


# In[ ]:


# Clustering seems to be adding no value after some trials and visualisation.


# ##### Analysing datewise relationship between firearms and drugs.

# In[ ]:


# A dataframe for month wise firearm.
df_firearm_month=df_joined.groupBy('Month').agg(cnt_cond(F.col('Crime_type').isin(['Possession of weapons', 'Public disorder and weapons'])).alias('firearm_monthwise'))
df_firearm_month.show(3)


# In[ ]:


# A dataframe for month wise drug incidents.
df_drugs_month=df_joined.groupBy('Month').agg(cnt_cond(F.col('Crime_type') =='Drugs').alias('drugs_monthwise'))
df_drugs_month.show(3)


# In[ ]:


# Joining drugs and firearms dataframes.
df_per_month= df_firearm_month.join(df_drugs_month, on='Month', how='left_outer')
df_per_month.show(3)


# In[ ]:


# Converting into pandas dataframe.
pdf_month = (
    df_per_month.select(
        "Month",
        "firearm_monthwise",
         "drugs_monthwise")
    .orderBy("Month")
    .toPandas()
)
pdf_month.head(3)


# In[ ]:


# Calculating rolling average for firearm by keeping window as 2.
pdf_month['rolling_firearm'] = pdf_month['firearm_monthwise'].rolling(2).mean().shift(-1)
pdf_month['rolling_firearm'].head(3)


# In[ ]:


# Calculating rolling average for drugs by keeping window as 2.
pdf_month['rolling_drugs'] = pdf_month['drugs_monthwise'].rolling(2).mean().shift(-1)
pdf_month['rolling_drugs'].head(3)


# In[ ]:


# Ploting monthwise variation for firearm vs drugs.
plt.figure(figsize=(9,6))
sns.lineplot(x="Month",y="firearm_monthwise",
             data=pdf_month,
             ci=None)
sns.lineplot(x="Month",y="rolling_firearm",
             data=pdf_month,
             ci=None)
sns.lineplot(x="Month",y="drugs_monthwise",
             data=pdf_month,
             ci=None)
sns.lineplot(x="Month",y="rolling_drugs",
             data=pdf_month,
             ci=None)
plt.legend(labels=['firearm_mothwise', 'rolling_firearm', 'drugs_monthwise', 'rolling_drugs'])
plt.title('monthwise firearm vs drugs')
plt.xlabel("Date", size=14)
plt.ylabel("number of crimes", size=14)


# ##### Validation of Claim-3: A correlation of 0.85 indicates that there is a strong ralationship between firearms and drugs. So, claim 3 is justified. Further, scatter plot makes the relationship clearly visible.

# <br/><br/>

# <font size='4'><b>Final thoughts:</b> Using 2018 dataset, we have accepted two hypothesis and rejected one. Generally speaking, rejected hypothesis (Birmingham case) is also somewhat true as it enjoys 29th position out of 309, but it can't be called as the highest. Other two hypothesis are accurate. Violent crime's gradual increase can't be inferred as Police Forces' failure alone, but can also be related to growing frustration. For this case, further research has to be made. All weapons are considered as firearms because of lack of publicly available data to dig deeper. Relationship between drugs and firearms is quite universal and only needs common sense to understand.</font>
