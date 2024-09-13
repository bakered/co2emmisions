#. %reset -f

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
#from plotly.subplots import make_subplots
#import copy
import time
import math


def weighted_percentile(values, weights, percentile): #values = plot_df[y_var][index_max_y]; weights=plot_df['pop'][index_max_y]; percentile=95
    """Compute the weighted percentile of a given list of values."""
    # Convert values and weights to numpy arrays
    values = np.array(values)
    weights = np.array(weights)
    # Remove NaNs from both values and weights
    mask = ~np.isnan(values) & ~np.isnan(weights)
    values = values[mask]
    weights = weights[mask]
    sorted_indices = np.argsort(values)
    sorted_values = np.array(values)[sorted_indices]
    sorted_weights = np.array(weights)[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    percentile_value = np.percentile(cumulative_weights, percentile)
    index = np.searchsorted(cumulative_weights, percentile_value)
    return sorted_values[index]

def createCountryBubbleGraph(datasource="GCP and Maddison",
                             geographyLevel='countries', 
                             x_var = 'gdp_per_capita', 
                             y_var = 'co2_per_capita', 
                             size_var = 'co2',
                             race_var = 'accumulated_co2',
                             smoothness = 2,
                             leave_trace = True, 
                             fixed_axes = True,
                             bubble_size=0,
                             flag_size= 1,
                             start_year=1950,
                             geography_list = [
                                 'ARG', 'AUS', 'BRA', 'CAN', 'CHN', 'FRA', 'DEU', 'IND', 'IDN', 
                                 'ITA', 'JPN', 'MEX', 'RUS', 'SAU', 'ZAF', 'KOR', 'TUR', 'GBR', 'USA'
                             ],
                             x_log=True,
                             y_log=False,
                             size_log=True,
                             show_flags=True,
                             start_time=time.time(),
                             progress=None,
                             ):
    
    #geographyLevel='countries'; x_var='gdp_per_capita'; y_var='co2_per_capita'; size_var='co2'; race_var='accumulated_co2'; leave_trace=True; fixed_axes=True; flag_size=1; x_log=False; y_log=False; show_flags=False; start_year=1950
    
    
    ########## SET PARAMETERS
    region1s = ["Developed", 
                "Developing Asia and Oceania", 
                "Latin America and the Carribean", 
                "Africa"]
    
    if False: #geographyLevel=='countries':
        color_map = {'Developed': '#009EDB', 'Developing': '#ED1847', 'LDCs': '#FFC800'}
        colour_var = 'region2'
    else:
        color_map = {'Developed': '#009EDB', 'Developing Asia and Oceania': '#ED1847', 'Latin America and the Carribean': '#72BF44', 'Africa': '#FFC800', }
        colour_var = 'region1'
    
    if datasource == "GCP and Maddison":
        labels = {"co2_per_capita": "CO2 per capita (Tons)",
                 "gdp_per_capita": "GDP per capita (2011 Dollars, PPP)",
                 "co2": "Yearly CO2 (Kilotons)",
                 "accumulated_co2": "Accumulated CO2 (Kilotons) (could be inconsistent!)", 
                 "pop": "Population", 
                 "gdp": "GDP (Millions)",
                 }
    else:
         labels = {"co2_per_capita": "CO2 per capita (Tons)",
                  "gdp_per_capita": "GDP per capita (2021 Dollars, PPP)",
                  "co2": "Yearly CO2 (Kilotons)",
                  "accumulated_co2": "Accumulated CO2 (Kilotons) (could be inconsistent!)", 
                  "pop": "Population", 
                  "gdp": "GDP (Millions)",
                  }
                  
    
    if geographyLevel == "countries":
        text_positions = {
            'USA': 'top left',
            "ZAF": 'top left', 
            "EGY": 'top left', 
            "DZA": 'top left', 
            "USA": 'top left', 
            "RUS": 'top left', 
            "JPN": 'top left', 
            "CHN": 'top left', 
            "IND": 'top left', 
            "IRN": 'top left', 
            "BRA": 'top left', 
            "MEX": 'top left', 
            "ARG": 'top left',
        }
    else:
        text_positions = {
            'Africa': 'middle right',
            'Developing Asia and Oceania': 'top left',
            'Latin America and the Carribean': 'bottom right',
            'Developed': 'bottom center'
        }
        
    if datasource == "GCP and Maddison":
        if start_year<1820:
            start_year = 1820
    else:
        if start_year<1990:
            start_year = 1990
    
    
    ########## LOAD DATA
    now_time = time.time()
    print(f"start function time: {now_time - start_time} seconds")
    progress.set(2, "Loading data...")
    
    # Import the data
    #data = pd.read_csv("dataCountries.csv")
    #data = pd.read_csv(open_url('https://github.com/bakered/co2emmisions/blob/main/src_shiny_app/dataPlot1.csv'))
    if datasource == "GCP and Maddison":
        if geographyLevel == "countries":
            infile = Path(__file__).parent / "dataCountries.csv"
        else:
            infile = Path(__file__).parent / "dataRegions.csv"
    else:
        if geographyLevel == "countries":
            infile = Path(__file__).parent / "dataWDICountries.csv"
        else:
            infile = Path(__file__).parent / "dataWDIRegions.csv"
    data = pd.read_csv(infile)
    data['region1'] = pd.Categorical(data['region1'], categories=region1s, ordered=True)
    data['year'] = data['year'].astype(int)
    
    now_time = time.time()
    print(f"loaded data time: {now_time - start_time} seconds")
    progress.set(3, "creating bubbles...")
    
    if geographyLevel == 'countries': 
        # create image link from ISO2
        data['ISO2']= data['ISO2'].astype(str)
        data['image_link'] = data['ISO2'].apply(lambda iso: f"https://hatscripts.github.io/circle-flags/flags/{iso.lower()}.svg")
        geography = 'ISO3'
    else:
        geography = 'region1'



    ########## FILTER AND ORDER DATA
    # Filter the DataFrame based on the list of ISO3 codes
    plot_df = data[(data[geography].isin(geography_list)) & (data['year'] > start_year)].copy()
    
    ##### reorder to prevent frantic swapping
    ## make changes to either scatter plot            
    desired_order = region1s 
    desired_order = [category for category in desired_order if category in plot_df[colour_var].unique()]
     
    # create ordering so NaNs come last
    mask = ~plot_df[[x_var, y_var, size_var]].isna().any(axis=1)
    iso3_counts = plot_df[mask][geography].value_counts().reset_index()
    iso3_counts.columns = [geography, 'count']  # Rename the columns
    plot_df = plot_df.merge(iso3_counts, on=geography)
    if geographyLevel == "countries":
        # create ordering such that the ISO3s for whom text should be written come first
        position_rank = {key: rank + 1 for rank, key in enumerate(text_positions.keys())}
        plot_df['custom_order'] = plot_df[geography].map(position_rank).fillna(len(position_rank) + 1).astype(int)
        plot_df = plot_df.sort_values(by=['region1', 'custom_order', 'count', geography, 'year'], ascending=[True, True, False, True, True])
        plot_df = plot_df.drop(columns=['count', 'custom_order'])
    else: 
        plot_df = plot_df.sort_values(by=['region1', 'count', 'year'], ascending=[True, False, True])
        plot_df = plot_df.drop(columns=['count'])

    #geography_list=['ARG', 'AUS', 'BRA', 'CAN', 'CHN', 'FRA', 'DEU', 'IND', 'IDN', 'ITA', 'JPN', 'MEX', 'RUS', 'SAU', 'ZAF', 'KOR', 'TUR', 'GBR', 'USA', 'SGP', 'PNG', 'MYS', 'BRN', 'IRN', 'OMN']
    
    
    
    
   ########## CALCULATE VALUES 
    # Calculate weighted 95% percentile
    index_max_x = plot_df.groupby(geography)[x_var].idxmax().dropna()
    index_max_y = plot_df.groupby(geography)[y_var].idxmax().dropna()
    max_x = weighted_percentile(plot_df[x_var][index_max_x], plot_df['pop'][index_max_x], 95) *1.2 # plot_df[x_var].max() * 1.2
    max_y = weighted_percentile(plot_df[y_var][index_max_y], plot_df['pop'][index_max_y], 95) *1.2 # plot_df[y_var].max() * 1.2
    min_x = plot_df[x_var].min() * 0.9
    min_y = plot_df[y_var].min() * 0.9
    

    # Calculate the range for logarithmic scale
    if x_log:
        max_x = np.log10(max_x) + np.log10(1.2)
        min_x = min_x if min_x > 0 else 0.1
        min_x = np.log10(min_x) - np.log10(1.2)
    else:
        min_x = 0
    if y_log:
        max_y = np.log10(max_y) + np.log10(1.2)
        min_y = min_y if min_y > 0 else 0.1
        min_y = np.log10(min_y) - np.log10(1.2)
    else:
        min_y = 0
        
    #deal with size of bubbles 
    plot_df['bubble_size'] = np.where(plot_df[size_var] > 0, plot_df[size_var] + bubble_size, plot_df[size_var])

    plot_df['bubble_size'] = plot_df['bubble_size'].fillna(0)
    scatter_size_max_parameter = 35

    
    if max_x>max_y:
        max_bubble_size_wanted = (max_x/10)*flag_size
    else:
        max_bubble_size_wanted = (max_y/5)*flag_size
    
    
    # Add normalized_size column such that the value is the diameter making 
    plot_df[size_var] = plot_df[size_var].fillna(0)
    plot_df.loc[:, 'normalized_size'] = 2 * np.sqrt(plot_df[size_var] /np.pi) 
    co2_max = plot_df['normalized_size'].max()
    # scale column such that the co2_max will be the size of the bubble you want measured in x-axis units. 
    # (warning: i think only works if x-axis is larger than y-axis)
    plot_df.loc[:, 'normalized_size'] *= (max_bubble_size_wanted / co2_max)
    
    #this is used later in flag addition
    geographies = plot_df[geography].unique()
    
    ########## ADD SMOOTHING
    progress.set(5, "Adding smoothness...")
    def expand_dataframe(df, n, geography, year, x_var, y_var, bubble_size, additional_cols):
        # Create an empty list to store the results
        expanded_rows = []
    
        # Iterate over unique geography values
        for geo in df[geography].unique():
            # Filter the dataframe for the current geography
            geo_df = df[df[geography] == geo].sort_values(by=year).reset_index(drop=True)
    
            # Iterate through consecutive rows
            for i in range(len(geo_df) - 1):
                # Get the starting and ending rows
                row_start = geo_df.iloc[i]
                row_end = geo_df.iloc[i + 1]
    
                # Create interpolated rows
                for j in range(n):
                    # Interpolation factor (j=0 gives the start row, j=n-1 gives the end row)
                    alpha = j / n
    
                    # Interpolate the numeric values using dynamic column names
                    interpolated_row = {
                        geography: row_start[geography],  # geography stays the same
                        year: row_start[year] * (1 - alpha) + row_end[year] * alpha,
                        x_var: row_start[x_var] * (1 - alpha) + row_end[x_var] * alpha,
                        y_var: row_start[y_var] * (1 - alpha) + row_end[y_var] * alpha,
                        bubble_size: row_start[bubble_size] * (1 - alpha) + row_end[bubble_size] * alpha
                    }
                   
                    # Keep additional columns (like image_link and region1) unchanged
                    for col in additional_cols:
                        interpolated_row[col] = row_start[col]
                        
                        
                    # Append the interpolated row to list of rows
                    expanded_rows.append(interpolated_row)

        
        expanded_df = pd.DataFrame(expanded_rows)
        return expanded_df


    if geographyLevel == "countries":
        additional_cols = ['image_link', 'region1']
    else:
        additional_cols = ['region1']
        
    plot_df = expand_dataframe(plot_df, 
                               n=smoothness, 
                               geography=geography, 
                               year='year', 
                               x_var=x_var, 
                               y_var=y_var, 
                               bubble_size='bubble_size',
                               additional_cols = additional_cols
                               )
    
    
    
    ########## CREATE SCATTER PLOT
    if geographyLevel == "countries":
        # Create the base plot with Plotly Express
        figScatter = px.scatter(
            plot_df, 
            x=x_var, 
            y=y_var, 
            color=colour_var,
            color_discrete_map=color_map,
            size='bubble_size', 
            text=geography,
            hover_name=geography,
            animation_frame='year', 
            size_max=scatter_size_max_parameter,
            template="plotly_white",
            #trendline="lowess"
        )
        
        if not x_log and not y_log:
            figScatter.update_traces(marker=dict(opacity=0.8))  # Set opacity to 0 for invisibility
        else:
            figScatter.update_traces(marker=dict(opacity=0.8))
            
        

    else:
        # Create the base plot with Plotly Express
        figScatter = px.scatter(
            plot_df, 
            x=x_var, 
            y=y_var, 
            color=colour_var,
            color_discrete_map=color_map,
            size='bubble_size', 
            text=geography,
            hover_name=geography,
            animation_frame='year', 
            size_max=scatter_size_max_parameter,
            template="plotly_white"
        )
        
        figScatter.update_traces(textposition='top right')
        figScatter.update_traces(marker=dict(opacity=0.8))
        
        
        
        
     #   print(figScatter.data)
      
    
    
    # Extract traces and reorder them
    ordered_traces = []
    for category in desired_order:
        # Filter traces for each category
        trace = next(tr for tr in figScatter.data if tr.name == category)
        ordered_traces.append(trace)        
    # Update the figure with reordered traces
    figScatter.data = ordered_traces
    figScatter.update_layout(
        legend_title_font=dict(size=16, family="Helvetica Neue LT Std 45 Light"),
        legend=dict(
            title="Region",
            x=0.10,            # x-coordinate of the legend (0 = left, 1 = right)
            y=0.85,            # y-coordinate of the legend (0 = bottom, 1 = top)
            xanchor='left', # Positioning relative to the x-coordinate (left)
            yanchor='top',  # Positioning relative to the y-coordinate (top)
            traceorder='normal',
            font = dict(family = "Helvetica Neue LT Std 45 Light", size = 15, color = "black"),
        )
    )

    # Extract color mapping from figLine
    #color_map = {trace.name: trace.marker.color for trace in figScatter.data if 'color' in trace.marker}
    
    # add log axes if necessary
    if x_log:
        figScatter.update_xaxes(type="log")
    
    if y_log:
        figScatter.update_yaxes(type='log')
        
    # following code does not work: it changes the hovertext on the first frame only, and in that case wrong... need to deal with frames!.
    #.. better to go trace by trace?
    
  #  figScatter.update_traces(
  #      customdata=plot_df[[geography, 'gdp']].values,
  #      hovertemplate="<br>".join([
  #          "<strong>%{customdata[0]}</strong><br>",
  #          "GDP (millions): %{customdata[1]:,.0f}",
  #          "<extra></extra>"  # Removes the trace name from hover
  #  ])
  #  )
    
    
    ########## CREATE LINE PLOT    
    now_time = time.time()
    print(f"Created scatter time: {now_time - start_time} seconds")
    progress.set(10, "calculating lines data...")
    
    if leave_trace:
        # xxxxx add a line of best fit : smooth curve..or pre-calculate and save in data?
        #  Loop over each year in the DataFrame
        expanded_rows = pd.DataFrame()
        for year in plot_df['year'].unique(): # year=plot_df['year'].unique()[12]
            # Filter the rows for the current year and all previous years
            filtered_rows = plot_df[plot_df['year'].astype(float) <= float(year)].copy()
            filtered_rows.loc[:, 'year'] = year
            # Append these rows to the list
            expanded_rows = pd.concat([expanded_rows, filtered_rows], axis=0, ignore_index=True)
        
        #print(expanded_rows[expanded_rows['ISO3'] =="JPN"])
        
        now_time = time.time()
        print(f"calculated lines time: {now_time - start_time} seconds")
        progress.set(40, "creating lines plot...")
        
        # Create an animated line plot
        figLine = px.line(
            expanded_rows, 
            x=x_var, 
            y=y_var, 
            color=colour_var,
            line_group=geography,
            color_discrete_map=color_map,
            #line_group=geography,  # Group by geography to draw a separate line for each
            #hover_name=geography, 
            animation_frame='year',
            template="plotly_white"
            )
        figLine.update_traces(hoverinfo='skip', hovertemplate=None)
        figLine.update_traces(opacity=0.5)

    

        # add log axes
        if x_log:
            figLine.update_xaxes(type="log")
        if y_log:
            figLine.update_yaxes(type='log')

        # Hide legend for figLine traces
        figLine.update_layout(showlegend=False)
        for trace in figLine.data:
            trace.showlegend = False
            
        ########### COMBINE PLOTS
        now_time = time.time()
        print(f"created line graph time: {now_time - start_time} seconds")
        progress.set(70, "Combining plots...")    
            
        # now integrate, traces, frames and layout
        fig = go.Figure(
          data=figLine.data + figScatter.data,
          frames=[
              go.Frame(data=fr1.data + fr2.data, name=fr2.name)
              for fr1, fr2 in zip(figLine.frames, figScatter.frames)
          ],
          layout=figScatter.layout,
        )
        #fig.update_layout(
        #    legend=dict(
        #        x=0,            # x-coordinate of the legend (0 = left, 1 = right)
        #        y=1,            # y-coordinate of the legend (0 = bottom, 1 = top)
        #        xanchor='left', # Positioning relative to the x-coordinate (left)
        #        yanchor='top',  # Positioning relative to the y-coordinate (top)
        #    )
        #)
        fig.update_traces(hoverinfo='skip', hovertemplate=None)
        
        now_time = time.time()
        print(f"combined graphs time: {now_time - start_time} seconds")
        progress.set(90, "finishing...")
        
    else:
        fig = figScatter
    
    
    ########## CHANGE CUSTOM SETTINGS
    progress.set(90, "custom settings...")
    #first set fig traces then go frame by frame and 
    # add flags or,
    # add labels
    # hovertext
    # set settings xxx combine all frame by frames
    
    num_bubbles = len(plot_df[geography].unique())
    

    # Update inital data 
    for trace in fig.data:
        #set frame data for all traces
        if trace.legendgroup == "Developed":
            trace.legendgroup = "Developed                            "
            trace.name = "Developed                            "
        elif trace.legendgroup == "Latin America and the Carribean":
            trace.legendgroup = "Latin America & Carribean"
            trace.name = "Latin America & Carribean"
        elif trace.legendgroup == "Developing Asia and Oceania":
            trace.legendgroup = "Developing Asia & Oceania"
            trace.name = "Developing Asia & Oceania"
        
        # set initial data for line traces
        if trace.mode == 'lines':
            trace.showlegend = False
        
        # set initial data for bubble data
        else:
            trace.showlegend = True
            # if fewer than 20 bubbles, then plot all names
            if num_bubbles <= 20:
                tracetextpositions = []
                for geography in trace.text:
                    if geography in text_positions:
                        tracetextpositions.append(text_positions[geography])
                    else:
                        tracetextpositions.append('top left')
                trace.textposition = tracetextpositions
                
            # if more than 20 bubbles then keep only the named geographies
            else:
                tracetext = []
                tracetextpositions = []
                for geography in trace.text:
                    if geography in text_positions:
                        tracetext.append(geography)
                        tracetextpositions.append(text_positions[geography])
                    else:
                        tracetext.append('')
                        tracetextpositions.append('top left')
                trace.text = tracetext
                trace.textposition = tracetextpositions
                
    
    # Update frame by frame
    for frame in fig.frames:
        #set traces in frame.data
        for trace in frame.data:
            #set frame data for all traces
            if trace.legendgroup == "Developed":
                trace.legendgroup = "Developed                            "
                trace.name = "Developed                            "
            elif trace.legendgroup == "Latin America and the Carribean":
                trace.legendgroup = "Latin America & Carribean"
                trace.name = "Latin America & Carribean"
            elif trace.legendgroup == "Developing Asia and Oceania":
                trace.legendgroup = "Developing Asia & Oceania"
                trace.name = "Developing Asia & Oceania"
            #print(frame.name)
            
            # set frame data for line traces
            if trace.mode == 'lines':
                trace.showlegend = False
            
            # set frame data for bubble traces
            else:
                trace.showlegend = True
                # if fewer than 20 bubbles, then plot all names
                if num_bubbles <= 20:
                    tracetextpositions = []
                    for geography in trace.text:
                        if geography in text_positions:
                            tracetextpositions.append(text_positions[geography])
                        else:
                            tracetextpositions.append('top left')
                    trace.textposition = tracetextpositions
                
                # if more than 20 bubbles keep only names geographies
                else:
                    tracetext = []
                    tracetextpositions = []
                    for geography in trace.text:
                        if geography in text_positions:
                            tracetext.append(geography)
                            tracetextpositions.append(text_positions[geography])
                        else:
                            tracetext.append('')
                            tracetextpositions.append('top left')
                    trace.text = tracetext
                    trace.textposition = tracetextpositions

        
        #set annotations in frame
        year = math.floor(float(frame.name))
        new_annotation = go.layout.Annotation(
            text=f"{year}",  # Annotation text showing the year
            showarrow=False,  # No arrow pointing to a data point
            xref="x",  # X position relative 'paper' or 'x' for 'data'
            yref="y",  # Y position relative 'paper' or 'y' for 'data'
            x=max_x * 0.95,  # X position (bottom right)
            y=max_y * 0.05,  # Y position (bottom right)
            font=dict(size=45, color="grey"),  # Font size and color
            align="right"  # Align text to the right
        )
        if 'annotations' in frame.layout:
            frame.layout.annotations += (new_annotation,)
        else:
            frame.layout.annotations = (new_annotation,)


    #change label in slider steps
    for step in fig.layout.sliders[0].steps:
        step.label = math.floor(float(step.label))
    
    full_index = pd.Index(geographies, name=geography)
    if geographyLevel == "countries" and not x_log and not y_log and show_flags:
        
        for frame in fig.frames: #frame = fig.frames[19]
            #print(frame.layout.images)
            year = frame.name
            #print(year)
            # Filter data for the specific year
            #if leave_trace:
            #  year_data = plot_df[plot_df['year'] <= floatyear)]
            #  ## add in NA rows for missing data
            #  full_years = list(range(1970, float(year)+1))
            #  full_index = pd.MultiIndex.from_product([countries, full_years], names=[geography, 'year'])
            #  year_data = year_data.set_index([geography, 'year']).reindex(full_index).reset_index()
            #else:
            year_data = plot_df[plot_df['year'] == year]
            ## add in NA rows for missing data
     
            
            year_data = year_data.set_index(geography).reindex(full_index).reset_index()
      
            # Create list of image annotations for this year
            image_annotations = []
            for i, row in year_data.iterrows():
              if pd.isna(row['image_link']):
                image_annotations.append(
                      {
                        'source': "https://hatscripts.github.io/circle-flags/flags/gb.svg",
                        'xref': "x",
                        'yref': "y",
                        'x': 0,
                        'y': 0,
                        'sizex': 0,
                        'sizey': 0,
                        'xanchor': "center",
                        'yanchor': "middle",
                        'sizing': "contain",
                        'opacity': 0.8,
                        'layer': "above"
                        }
                      )
              else:
                image_annotations.append(
                  {
                          'source': row['image_link'],
                          'xref': "x",
                          'yref': "y",
                          'x': row[x_var],
                          'y': row[y_var],
                          'sizex': row['normalized_size'],
                          'sizey': row['normalized_size'],
                          'xanchor': "center",
                          'yanchor': "middle",
                          'sizing': "contain",
                          'opacity': 0.8,
                          'layer': "above"
                          }
                      )
            
            
            
        now_time = time.time()
        print(f"added flags time: {now_time - start_time} seconds")
        progress.set(95, "printing plot...")
     
        
    

    ######## LABELS, AXES, AND SLIDER
    #set some parameters
    y_var_label = labels.get(y_var, y_var)  # Default to y_var if not found in labels
    x_var_label = labels.get(x_var, x_var)  # Default to x_var if not found in labels
    buttons = [{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 400/smoothness, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate', 'transition': {'duration': 400/smoothness}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 400/smoothness, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 400/smoothness}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]
    sliders=[{
        'active': 0,
        #'currentvalue': {'prefix': 'Frame: '},
        'x': 0.1,
        'len': 0.9,
        'pad': {'b': 10},
    }]
    
    
    # Adjust layout
    fig.update_layout(
        xaxis_title=x_var_label,
        yaxis_title=y_var_label,
        title={'text': f"<span style='font-size:28px; font-family:Helvetica Neue LT Std 45 Light;'><b>The glaring inequality of income and CO2 emissions</b></span><br><span style='font-size:18px;'><i>{y_var_label} vs {x_var_label}</i></span>",
               'x': 0.08,
               'xanchor': 'left',},
        autosize=True,
        #width=1100,
        #height=700,
        updatemenus=buttons,
        #sliders=sliders,
        title_font=dict(family="Helvetica Neue LT Std 45 Light", size=24, color="black"),
        xaxis_title_font=dict(family="Helvetica Neue LT Std 45 Light", size=18, color="black"),
        yaxis_title_font=dict(family="Helvetica Neue LT Std 45 Light", size=18, color="black"),
        font=dict(family="Helvetica Neue LT Std 45 Light", size=14, color="black"),
    )
    
    if fixed_axes:
        fig.update_yaxes(fixedrange = True)
        fig.update_xaxes(fixedrange = True)
        fig.update_layout(
          xaxis=dict(range=[min_x, max_x]),
          yaxis=dict(range=[min_y, max_y])
          )

    
    ######### SAVE OR RETURN PLOT
    now_time = time.time()
    print(f"plot created time: {now_time - start_time} seconds")
    
   # fig.write_html('/Users/edbaker/UN_projects/c02emmisions/plotly_animation.html')
    
    now_time = time.time()
    print(f"plot saved time: {now_time - start_time} seconds")
    
    
    #print(figScatter.layout)
    #print(fig.data)
    # print(fig.layout)
    
    return(fig)
    
#fig = createCountryBubbleGraph()    
# Save the figure as HTML
#fig.write_html('/Users/edbaker/UN_projects/c02emmisions/plotly_animation.html')
