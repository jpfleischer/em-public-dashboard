import pandas as pd
import numpy as np

import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.timeseries.tcquery as esttc
import emission.core.wrapper.localdate as ecwl

# Module for pretty-printing outputs (e.g. head) to help users
# understand what is going on
# However, this means that this module can only be used in an ipython notebook

import IPython.display as disp

import emission.core.get_database as edb

def get_time_query(year, month):
    if year is None and month is None:
        return None

    if month is None:
        assert year is not None
        query_ld = ecwl.LocalDate({"year": year})
    else:
        assert year is not None and month is not None
        query_ld = ecwl.LocalDate({"year": year, "month": month})
    tq = esttc.TimeComponentQuery("data.start_local_dt", query_ld, query_ld)
    return tq

def get_participant_uuids(program):
    """
        Get the list of non-test users in the current program.
        Note that the "program" parameter is currently a NOP and should be removed in
        conjunction with modifying the notebooks.
    """
    all_users = pd.json_normalize(edb.get_uuid_db().find())
    participant_list = all_users[np.logical_not(all_users.user_email.str.contains("_test_"))]
    participant_uuid_str = participant_list.uuid
    disp.display(participant_list.user_email)
    return participant_uuid_str

def load_all_confirmed_trips(tq):
    agg = esta.TimeSeries.get_aggregate_time_series()
    all_ct = agg.get_data_df("analysis/confirmed_trip", tq)
    print("Loaded all confirmed trips of length %s" % len(all_ct))
    disp.display(all_ct.head())
    return all_ct

def load_all_participant_trips(program, tq):
    participant_list = get_participant_uuids(program)
    all_ct = load_all_confirmed_trips(tq)
    participant_ct_df = all_ct[all_ct.user_id.isin(participant_list)]
    print("After filtering, found %s participant trips " % len(participant_ct_df))
    disp.display(participant_ct_df.head())
    return participant_ct_df

def filter_labeled_trips(mixed_trip_df):
    labeled_ct = mixed_trip_df[mixed_trip_df.user_input != {}]
    print("After filtering, found %s labeled trips" % len(labeled_ct))
    disp.display(labeled_ct.head())
    return labeled_ct

def expand_userinputs(labeled_ct, labels_per_trip):
    '''
    param: labeled_ct: a dataframe of confirmed trips, some of which have labels
    params: labels_per_trip: the number of labels for each trip.
        Currently, this is 2 for studies and 3 for programs, and should be 
        passed in by the notebook based on the input config.
        If used with a trip-level survey, it could be even larger.
    '''
    label_only = pd.DataFrame(labeled_ct.user_input.to_list(), index=labeled_ct.index)
    disp.display(label_only.head())
    expanded_ct = pd.concat([labeled_ct, label_only], axis=1)
    assert len(expanded_ct) == len(labeled_ct), \
        ("Mismatch after expanding labels, expanded_ct.rows = %s != labeled_ct.rows %s" %
            (len(expanded_ct), len(labeled_ct)))
    print("After expanding, columns went from %s -> %s" %
        (len(labeled_ct.columns), len(expanded_ct.columns)))
    assert len(expanded_ct.columns) == len(labeled_ct.columns) + labels_per_trip, \
        ("Mismatch after expanding labels, expanded_ct.columns = %s != labeled_ct.columns %s" %
            (len(expanded_ct.columns), len(labeled_ct.columns)))
    disp.display(expanded_ct.head())
    return expanded_ct

def load_viz_notebook_data(year, month, program, dic_re, df_ei=None, dic_pur=None, dic_fuel=None):
    """ Inputs:
    year/month/program = parameters from the visualization notebook
    dic/df_* = label mappings; if included they will be used to recode user inputs
    
    Pipeline to load and process the data before use in visualization notebooks.
    """
    # Access database
    tq = get_time_query(year, month)
    participant_ct_df = load_all_participant_trips(program, tq)
    labeled_ct = filter_labeled_trips(participant_ct_df)
    expanded_ct = expand_userinputs(labeled_ct, 3)
    expanded_ct = data_quality_check(expanded_ct)

    # Change meters to miles
    unit_conversions(expanded_ct)

    # Mapping new labels with dictionaries, calculating energy impact
    expanded_ct['Mode_confirm']= expanded_ct['mode_confirm'].map(dic_re)
    expanded_ct['Replaced_mode']= expanded_ct['replaced_mode'].map(dic_re)
    if dic_pur is not None:
        expanded_ct['Trip_purpose']= expanded_ct['purpose_confirm'].map(dic_pur)
    if dic_fuel is not None:
        expanded_ct['Mode_confirm_fuel']= expanded_ct['Mode_confirm'].map(dic_fuel)
        expanded_ct['Replaced_mode_fuel']= expanded_ct['Replaced_mode'].map(dic_fuel)
    if df_ei is not None:
        expanded_ct = energy_intensity(expanded_ct, df_ei, 'distance_miles', 'Replaced_mode', 'Mode_confirm')
        expanded_ct = energy_impact_kWH(expanded_ct, 'distance_miles', 'Replaced_mode', 'Mode_confirm')
        expanded_ct = CO2_impact_lb(expanded_ct, 'distance_miles', 'Replaced_mode', 'Mode_confirm')

    # Document data quality
    file_suffix = get_file_suffix(year, month, program)
    quality_text = get_quality_text(participant_ct_df, expanded_ct)

    return expanded_ct, file_suffix, quality_text

def get_quality_text(participant_ct_df, expanded_ct):
    cq = (len(expanded_ct), len(expanded_ct.user_id.unique()), len(participant_ct_df), len(participant_ct_df.user_id.unique()), (len(expanded_ct) * 100) / len(participant_ct_df), )
    quality_text = "Based on %s confirmed trips from %d users\nof %s total trips from %d users (%.2f%%)" % cq
    print(quality_text)
    return quality_text

def get_file_suffix(year, month, program):
    suffix = "_%04d" % year if year is not None else ""
    suffix = suffix + "_%02d" % month if month is not None else ""
    suffix = suffix + "_%s" % program if program is not None else ""
    print(suffix)
    return suffix

def access_alt_text(alt_text, chart_name):
    """ Inputs:
    alt_text = the text describing the chart
    chart_name = the alt text file to save or update
    """
    f = open(f'/plots/{chart_name}.txt','w')
    f.write(alt_text)
    f.close()
    return alt_text

def store_alt_text_generic(chart_description, chart_name, var_name):
    """ Inputs:
    chart_description = what type of chart is it
    chart_name = what to label chart by in the dictionary
    var_name = the variable being analyzed across the chart
    """
    # Fill out the alt text based on components of the chart and passed data
    alt_text = f"{chart_description} chart of {var_name}."
    alt_text = access_alt_text(alt_text, chart_name)
    return alt_text

def store_alt_text_bar(df, chart_name, var_name):
    """ Inputs:
    df = dataframe with index of item names, first column is counts
    chart_name = what to label chart by in the dictionary
    var_name = the variable being analyzed across pie slices
    """
    # Fill out the alt text based on components of the chart and passed data
    alt_text = f"Bar chart of {var_name}."
    for i in range(0,len(df)):
        alt_text += f" {df.index[i]} is {np.round(df.iloc[i,0], 1)}."
    alt_text = access_alt_text(alt_text, chart_name)
    return alt_text

def store_alt_text_pie(df, chart_name, var_name):
    """ Inputs:
    df = dataframe with index of item names, first column is counts
    chart_name = what to label chart by in the dictionary
    var_name = the variable being analyzed across pie slices
    """
    # Fill out the alt text based on components of the chart and passed data
    alt_text = f"Pie chart of {var_name}."
    for i in range(0,len(df)):
        alt_text += f" {df.index[i]} is {np.round(df.iloc[i,0] / np.sum(df.iloc[:,0]) * 100, 1)}%."
    alt_text = access_alt_text(alt_text, chart_name)
    return alt_text

def store_alt_text_timeseries(df, chart_name, var_name):
    """ Inputs:
    df = dataframe with first col of dates, second column is values
    chart_name = what to label chart by in the dictionary
    var_name = the variable being analyzed across pie slices
    """
    # Fill out the alt text based on components of the chart and passed data
    alt_text = f"Scatter chart of {var_name}."
    arg_min = np.argmin(df.iloc[:,1])
    arg_max = np.argmax(df.iloc[:,1])
    alt_text += f" First minimum is {np.round(df.iloc[arg_min,1], 1)} on {df.iloc[arg_min,0]}. First maximum is {np.round(df.iloc[arg_max,1], 1)} on {df.iloc[arg_max,0]}."
    alt_text = access_alt_text(alt_text, chart_name)
    return alt_text

def data_quality_check(expanded_ct):
    '''1. Delete rows where the mode_confirm was pilot_ebike and repalced_mode was pilot_ebike.
       2. Delete rows where the mode_confirm was pilot_ebike and repalced_mode was same_mode.
       3. Replace same_mode for the mode_confirm for Energy Impact Calcualtion.'''

    # TODO: This is only really required for the initial data collection around the minipilot
    # in subsequent deployes, we removed "same mode" and "pilot_ebike" from the options, so the
    # dataset did not contain of these data quality issues

    if 'replaced_mode' in expanded_ct.columns:
        expanded_ct.drop(expanded_ct[(expanded_ct['mode_confirm'] == 'pilot_ebike') & (expanded_ct['replaced_mode'] == 'pilot_ebike')].index, inplace=True)
        expanded_ct.drop(expanded_ct[(expanded_ct['mode_confirm'] == 'pilot_ebike') & (expanded_ct['replaced_mode'] == 'same_mode')].index, inplace=True)
        expanded_ct['replaced_mode'] = np.where(expanded_ct['replaced_mode'] == 'same_mode',expanded_ct['mode_confirm'], expanded_ct['replaced_mode'])
    
    return expanded_ct

def unit_conversions(df):
    df['distance_miles']= df["distance"]*0.00062 #meters to miles

def energy_intensity(df,df1,distance,col1,col2):
    """ Inputs:
    df = dataframe with data
    df = dataframe with energy factors
    distance = distance in meters
    col1 = Replaced_mode
    col2= Mode_confirm

    """
    df1 = df1.copy()
    df1[col1] = df1['mode']
    dic_ei_factor = dict(zip(df1[col1],df1['energy_intensity_factor']))
    dic_CO2_factor = dict(zip(df1[col1],df1['CO2_factor']))
    dic_ei_trip = dict(zip(df1[col1],df1['(kWH)/trip']))
    
    df['ei_'+col1] = df[col1].map(dic_ei_factor)
    df['CO2_'+col1] = df[col1].map(dic_CO2_factor)
    df['ei_trip_'+col1] = df[col1].map(dic_ei_trip)
    
      
    df1[col2] = df1[col1]
    dic_ei_factor = dict(zip(df1[col2],df1['energy_intensity_factor']))
    dic_ei_trip = dict(zip(df1[col2],df1['(kWH)/trip']))
    dic_CO2_factor = dict(zip(df1[col2],df1['CO2_factor']))
    df['ei_'+col2] = df[col2].map(dic_ei_factor)
    df['CO2_'+col2] = df[col2].map(dic_CO2_factor)
    df['ei_trip_'+col2] = df[col2].map(dic_ei_trip)
           
    return df


def energy_impact_kWH(df,distance,col1,col2):
    """ Inputs:
    df = dataframe with data
    distance = distance in miles
    col1 = Replaced_mode
    col2= Mode_confirm
    """
        
    conditions_col1 = [(df['Replaced_mode_fuel'] =='gasoline'),
                       (df['Replaced_mode_fuel'] == 'diesel'),
                       (df['Replaced_mode_fuel'] == 'electric')]
   
    conditions_col2 = [(df['Mode_confirm_fuel'] =='gasoline'),
                       (df['Mode_confirm_fuel'] == 'diesel'),
                       (df['Mode_confirm_fuel'] == 'electric')]

    gasoline_col1 = (df[distance]*df['ei_'+col1]*0.000293071) # 1 BTU = 0.000293071 kWH
    diesel_col1   = (df[distance]*df['ei_'+col1]*0.000293071)
    electric_col1 = (df[distance]*df['ei_'+col1])+ df['ei_trip_'+col1]
    
    gasoline_col2 = (df[distance]*df['ei_'+col2]*0.000293071)
    diesel_col2   = (df[distance]*df['ei_'+col2]*0.000293071)
    electric_col2 = (df[distance]*df['ei_'+col2])+ df['ei_trip_'+col2]
  
    
    values_col1 = [gasoline_col1,diesel_col1,electric_col1]
    values_col2 = [gasoline_col2,diesel_col2,electric_col2]  
    
    df[col1+'_EI(kWH)'] = np.select(conditions_col1, values_col1)
    df[col2+'_EI(kWH)'] = np.select(conditions_col2, values_col2)
    
    df['Energy_Impact(kWH)']  = round((df[col1+'_EI(kWH)'] - df[col2+'_EI(kWH)']),3)
  
    return df


def CO2_impact_lb(df,distance,col1,col2):
    """ Inputs:
    df = dataframe with data
    distance = distance in miles
    col1 = Replaced_mode
    col2= Mode_confirm
    """
 
    conditions_col1 = [(df['Replaced_mode_fuel'] =='gasoline'),
                       (df['Replaced_mode_fuel'] == 'diesel'),
                       (df['Replaced_mode_fuel'] == 'electric')]
   
    conditions_col2 = [(df['Mode_confirm_fuel'] =='gasoline'),
                       (df['Mode_confirm_fuel'] == 'diesel'),
                       (df['Mode_confirm_fuel'] == 'electric')]

  
    gasoline_col1 = (df[distance]*df['ei_'+col1]*0.000001)* df['CO2_Replaced_mode']
    diesel_col1   = (df[distance]*df['ei_'+col1]*0.000001)* df['CO2_Replaced_mode']
    electric_col1 = (((df[distance]*df['ei_'+col1])+df['ei_trip_'+col1])*0.001)*df['CO2_'+col1]
    
    gasoline_col2 = (df[distance]*df['ei_'+col2]*0.000001)* df['CO2_Mode_confirm']
    diesel_col2   = (df[distance]*df['ei_'+col2]*0.000001)* df['CO2_Mode_confirm']
    electric_col2 = (((df[distance]*df['ei_'+col2])+df['ei_trip_'+col2])*0.001)*df['CO2_'+col2]
  
    
    values_col1 = [gasoline_col1,diesel_col1,electric_col1]
    values_col2 = [gasoline_col2,diesel_col2,electric_col2]  
    
    df[col1+'_lb_CO2'] = np.select(conditions_col1, values_col1)
    df[col2+'_lb_CO2'] = np.select(conditions_col2, values_col2)
    df['CO2_Impact(lb)']  = round((df[col1+'_lb_CO2'] - df[col2+'_lb_CO2']),3)
  
    return df