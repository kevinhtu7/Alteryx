#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install mysql-connector-python


# In[3]:


pip install dask


# In[4]:


from sqlalchemy import create_engine
import pandas as pd
import dask.dataframe as dd
import time
import os


# In[5]:


MYSQL_HOST = 'heratdevisha.lmu.build'
MYSQL_USER = 'heratdev_admin_alteryx_capstone'
MYSQL_PASSWORD = 'AlteryxCapstone2024'
MYSQL_DB = 'heratdev_alteryx_capstone' 

lmubuildengine = create_engine(f'mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}')
aws_uri = f'mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}'


# In[9]:


access_level = """
    SELECT access_levels
    FROM Users
    JOIN Roles
    ON Users.role = Roles.role
    WHERE UserID = 'seal_kala';
           """


# In[10]:


access_level = pd.read_sql_query(access_level, con=lmubuildengine)
access_level

