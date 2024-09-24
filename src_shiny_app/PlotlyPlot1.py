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
import statsmodels.api as sm

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
                             bubble_similarity=0,
                             flag_size= 1,
                             bubble_size = 1,
                             rolling_mean_years=10,
                             start_year=1950,
                             geography_list = [
                                 'ARG', 'AUS', 'BRA', 'CAN', 'CHN', 'FRA', 'DEU', 'IND', 'IDN', 
                                 'ITA', 'JPN', 'MEX', 'RUS', 'SAU', 'ZAF', 'KOR', 'TUR', 'GBR', 'USA'
                             ],
                             x_log=True,
                             y_log=False,
                             size_log=True,
                             show_flags=True,
                             use_loess=True,
                             start_time=time.time(),
                             progress=None,
                             ):
    
    #geographyLevel='countries'; x_var='gdp_per_capita'; y_var='co2_per_capita'; size_var='co2'; race_var='accumulated_co2'; leave_trace=True; fixed_axes=True; flag_size=1; x_log=False; y_log=False; show_flags=False; start_year=1950
    
    
    ########## SET PARAMETERS
    region1s = ["Developed", 
                "Developing Asia and Oceania", 
                "Latin America and the Caribbean", 
                "Africa"]
    
    if False: #geographyLevel=='countries':
        color_map = {'Developed': '#009EDB', 'Developing': '#ED1847', 'LDCs': '#FFC800'}
        colour_var = 'region2'
    else:
        color_map = {'Developed': '#009EDB', 'Developing Asia and Oceania': '#ED1847', 'Latin America and the Caribbean': '#72BF44', 'Africa': '#FFC800', 
                     'Developing Asia <br>and Oceania': '#ED1847', 'Latin America <br>and the Caribbean': '#72BF44',}
        colour_var = 'region1'
    
    if datasource == "GCP and Maddison":
        labels = {"co2_per_capita": "CO<sub>2</sub> per capita",
                 "gdp_per_capita": "GDP per capita",
                 "co2": "Yearly CO<sub>2</sub>",
                 "accumulated_co2": "Accumulated CO<sub>2</sub>", 
                 "pop": "Population", 
                 "gdp": "GDP",
                 }
        labels_parenthesis = {"co2_per_capita": "<br>(Tons)",
                 "gdp_per_capita": "<br>(2011 Dollars, PPP)",
                 "co2": "(Kilotons)",
                 "accumulated_co2": "<br>(Kilotons) (could be inconsistent!)", 
                 "pop": "", 
                 "gdp": "<br>(Millions)",
                 }
    else:
         labels = {"co2_per_capita": "CO<sub>2</sub> per capita",
                  "gdp_per_capita": "GDP per capita",
                  "co2": "Yearly CO<sub>2</sub>",
                  "accumulated_co2": "Accumulated CO<sub>2</sub>", 
                  "pop": "Population", 
                  "gdp": "GDP",
                  }
         labels_parenthesis = {"co2_per_capita": " (Tons)",
                  "gdp_per_capita": " (2021 Dollars, PPP)",
                  "co2": " (Kilotons)",
                  "accumulated_co2": " (Kilotons) (could be inconsistent!)", 
                  "pop": "", 
                  "gdp": " (Millions)",
                  }
                  
    
    if geographyLevel == "countries":
        text_positions = {
            'USA': 'top right',
            "ZAF": 'top right', 
            "EGY": 'top right', 
            "DZA": 'top right', 
            "USA": 'top right', 
            "RUS": 'top right', 
            "JPN": 'top right', 
            "CHN": 'top right', 
            "IND": 'top right', 
            "IRN": 'top right', 
            "BRA": 'top right', 
            "MEX": 'top right', 
            "ARG": 'top right',
        }
    else:
        text_positions = {
            'Africa': 'middle right',
            'Developing Asia and Oceania': 'top center',
            'Latin America and the Caribbean': 'bottom right',
            'Developed': 'bottom center'
        }
        
    if datasource == "GCP and Maddison":
        if start_year<1820:
            start_year = 1820
    else:
        if start_year<1990:
            start_year = 1990
    
    #set some labels
    if x_log: 
        x_log_label = " in logs"
    else:
        x_log_label = ""
    if y_log: 
        y_log_label = " in logs"
    else:
        y_log_label = ""
    x_var_label = "<b>" + labels.get(x_var, x_var) + x_log_label + labels_parenthesis.get(x_var) + "</b>" # 
    y_var_label = "<b>" + labels.get(y_var, y_var) + y_log_label + labels_parenthesis.get(y_var) + "</b>" # 
    
    
    
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
        region_to_image_link = {
            "Africa": "https://raw.githubusercontent.com/bakered/co2emmisions/main/src_shiny_app/africa_map.png",
            "Developed": "https://raw.githubusercontent.com/bakered/co2emmisions/main/src_shiny_app/developed_map.png",
            "Developing Asia and Oceania": "https://raw.githubusercontent.com/bakered/co2emmisions/main/src_shiny_app/asia_and_oceania_map.png",
            "Latin America and the Caribbean": "https://raw.githubusercontent.com/bakered/co2emmisions/main/src_shiny_app/latin_america_and_the_caribbean_map.png"
        }

        # Use map() to create the new column 'image_link'
        data['image_link'] = data['region1'].map(region_to_image_link)
        


    ########## FILTER AND ORDER DATA
    latin_america_countries = ["ABW", "AIA", "ARG", "ATG", "BES", "BHS", "BLZ", "BOL", "BRA", 
                               "BRB", "CHL", "COL", "CRI", "CUB", "CUW", "DMA", "DOM", "ECU", 
                               "GRD", "GTM", "GUY", "HND", "HTI", "JAM", "LCA", "MEX", "MSR", 
                               "NIC", "PAN", "PER", "PRY", "SLV", "SUR", "SXM", "TCA", "TTO", 
                               "URY", "VCT", "VEN", "VGB"]
    african_countries = ["AGO", "BDI", "BEN", "BFA", "BWA", "CAF", "CIV", "CMR", "COD",
                         "COG", "COM", "CPV", "DJI", "DZA", "EGY", "ERI", "ETH", "GAB", 
                         "GHA", "GIN", "GMB", "GNB", "GNQ", "KEN", "LBR", "LBY", "LSO", 
                         "MAR", "MDG", "MLI", "MOZ", "MRT", "MUS", "MWI", "NAM", "NER", 
                         "NGA", "RWA", "SDN", "SEN", "SHN", "SLE", "SOM", "SSD", "STP", 
                         "SWZ", "SYC", "TCD", "TGO", "TUN", "TZA", "UGA", "ZAF", "ZMB", 
                         "ZWE"]
    asia_countries = ["AFG", "ARE", "ARM", "AZE", "BGD", "BHR", "BRN", "BTN", "CHN", 
                      "COK", "FJI", "FSM", "GEO", "HKG", "IDN", "IND", "IRN", "IRQ", 
                      "JOR", "KAZ", "KGZ", "KHM", "KIR", "KWT", "LAO", "LBN", "LKA", 
                      "MAC", "MDV", "MHL", "MMR", "MNG", "MYS", "NCL", "NIU", "NPL", 
                      "NRU", "OMN", "PAK", "PHL", "PLW", "PNG", "PRK", "PSE", "PYF", 
                      "QAT", "SAU", "SGP", "SLB", "SYR", "THA", "TJK", "TKM", "TLS", 
                      "TON", "TUR", "TUV", "TWN", "UZB", "VNM", "VUT", "WLF", "WSM", 
                      "YEM"]
    developed_countries = ["ALB", "AND", "AUS", "AUT", "BEL", "BGR", "BIH", "BLR", "BMU", 
                           "CAN", "CHE", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN", 
                           "FRA", "FRO", "GBR", "GRC", "GRL", "HRV", "HUN", "IRL", "ISL", 
                           "ISR", "ITA", "JPN", "KOR", "LIE", "LTU", "LUX", "LVA", "MDA", 
                           "MKD", "MLT", "MNE", "NLD", "NOR", "NZL", "POL", "PRT", "ROU", 
                           "RUS", "SPM", "SRB", "SVK", "SVN", "SWE", "UKR", "USA"]
    

    geography_list = list(geography_list)
    
    if geographyLevel == "countries":
        if "Latin America and the Caribbean" in geography_list:
            geography_list.remove("Latin America and the Caribbean")  # Modify in place
            geography_list += latin_america_countries
            
        if "Africa" in geography_list:
            geography_list.remove("Africa")
            geography_list += african_countries
            
        if "Developed" in geography_list:
            geography_list.remove("Developed")
            geography_list += developed_countries
            
        if "Developing Asia and Oceania" in geography_list:
            geography_list.remove("Developing Asia and Oceania")
            geography_list += asia_countries
        
        geography_list = list(set(geography_list))

    
    # Flatten the list in case it contains sublists
    geography_list = [item for sublist in geography_list for item in (sublist if isinstance(sublist, list) else [sublist])]

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
    max_x = weighted_percentile(plot_df[x_var][index_max_x], plot_df['pop'][index_max_x], 95) *2.5 # plot_df[x_var].max() * 1.2
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
    plot_df['bubble_size'] = np.where(plot_df[size_var] > 0, plot_df[size_var] + bubble_similarity, plot_df[size_var])

    plot_df['bubble_size'] = plot_df['bubble_size'].fillna(0)
    scatter_size_max_parameter = 60 * bubble_size

    # set parameter for flags
    if max_x>max_y:
        if geographyLevel == "countries":
            max_bubble_size_wanted = (max_x/24)*flag_size
        else:
            max_bubble_size_wanted = (max_x/40)*flag_size
    else:
        if geographyLevel == "countries":
            max_bubble_size_wanted = (max_y/12)*flag_size
        else:
            max_bubble_size_wanted = (max_y/20)*flag_size
    
    
    # Add normalised_size column such that the value is the diameter making 
    plot_df.loc[:, 'normalised_size'] = 2 * np.sqrt(plot_df['bubble_size'] /np.pi) 
    co2_max = plot_df['normalised_size'].max()
    # scale column such that the co2_max will be the size of the bubble you want measured in x-axis units. 
    # (warning: i think only works if x-axis is larger than y-axis)
    plot_df.loc[:, 'normalised_size'] *= (max_bubble_size_wanted / co2_max)
    
    
    #this is used later in flag addition
    geographies = plot_df[geography].unique()
    
    ########## ADD SMOOTHING
    progress.set(5, "Adding smoothness...")
    def expand_dataframe(df, n, geography, year, x_var, y_var, bubble_size, normalised_size, additional_cols):
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
                for j in range(n+1):
                    # Interpolation factor (j=0 gives the start row, j=n-1 gives the end row)
                    alpha = j / n
    
                    # Interpolate the numeric values using dynamic column names
                    interpolated_row = {
                        geography: row_start[geography],  # geography stays the same
                        year: row_start[year] * (1 - alpha) + row_end[year] * alpha,
                        x_var: row_start[x_var] * (1 - alpha) + row_end[x_var] * alpha,
                        y_var: row_start[y_var] * (1 - alpha) + row_end[y_var] * alpha,
                        bubble_size: row_start[bubble_size] * (1 - alpha) + row_end[bubble_size] * alpha,
                        normalised_size: row_start[normalised_size] * (1 - alpha) + row_end[normalised_size] * alpha,
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
        additional_cols = ['image_link', 'region1']
        
    plot_df = expand_dataframe(plot_df, 
                               n=smoothness, 
                               geography=geography, 
                               year='year', 
                               x_var=x_var, 
                               y_var=y_var, 
                               bubble_size='bubble_size',
                               normalised_size = 'normalised_size',
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
            figScatter.update_traces(marker=dict(opacity=1))  # Set opacity to 0 for invisibility
        else:
            figScatter.update_traces(marker=dict(opacity=1))
            
        

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
        figScatter.update_traces(marker=dict(opacity=1))
        
        
        
        
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
        legend_title_font=dict(size=16, family="Inter"),
        legend=dict(
            title="",
            orientation="h",
            entrywidth=150,
           # entrywidthmode='fraction',
            yanchor="bottom",
            y=1,
            xanchor="center",
            x=0.5,
            #x=0.10,            # x-coordinate of the legend (0 = left, 1 = right)
            #y=0.85,            # y-coordinate of the legend (0 = bottom, 1 = top)
            #xanchor='left', # Positioning relative to the x-coordinate (left)
            #yanchor='top',  # Positioning relative to the y-coordinate (top)
            #traceorder='normal',
            font = dict(family = "Inter", size = 16, color = "black"),
        ),
        plot_bgcolor='#F4F9FD',  # Background color inside the axes
        paper_bgcolor='#F4F9FD',
        xaxis=dict(
            linecolor='black',  # Color of the x-axis line
            linewidth=2,        # Thickness of the x-axis line
        ),
        yaxis=dict(
            linecolor='black',  # Color of the y-axis line
            linewidth=2,        # Thickness of the y-axis line
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

       


        if use_loess:
            def loess_smoothing(x, y, frac=0.3):
                # Apply LOESS (Locally Weighted Scatterplot Smoothing)
                loess_model = sm.nonparametric.lowess(y, x, frac=frac)
                return loess_model[:, 1]
            
            # Group by 'geography' and apply LOESS smoothing to 'y_var' over 'year'
            plot_df_line = pd.DataFrame()
            for geog, group in plot_df.groupby(geography):
                # Apply LOESS smoothing for y_var (you can adjust frac for smoother/less smooth fit)
                group[x_var] = loess_smoothing(group['year'], group[x_var], frac=0.3)
                group[y_var] = loess_smoothing(group['year'], group[y_var], frac=0.3)
                # Append smoothed data to a new DataFrame
                plot_df_line = pd.concat([plot_df_line, group], axis=0)
        else:
            
            plot_df_line = pd.DataFrame()
            for geog, group in plot_df.groupby(geography):
                group[x_var] = group[x_var].rolling(rolling_mean_years).mean()
                group[y_var] = group[y_var].rolling(rolling_mean_years).mean()

                
                plot_df_line = pd.concat([plot_df_line, group], axis=0)
            
        
        expanded_rows = pd.DataFrame()
        for year in plot_df_line['year'].unique(): # year=plot_df['year'].unique()[12]
            # Filter the rows for the current year and all previous years
            filtered_rows = plot_df_line[plot_df_line['year'].astype(float) <= float(year)].copy()
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
        figLine.update_traces(line={'width': 5})
        figLine.update_traces(hoverinfo='skip', hovertemplate=None)
        figLine.update_traces(opacity=0.8)

    

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
            trace.legendgroup = "Developed"
            trace.name = "Developed"
        elif trace.legendgroup == "Latin America and the Caribbean":
            trace.legendgroup = "Latin America <br>and the Caribbean"
            trace.name = "Latin America <br>and the Caribbean"
        elif trace.legendgroup == "Developing Asia and Oceania":
            trace.legendgroup = "Developing Asia <br>and Oceania"
            trace.name = "Developing Asia <br>and Oceania"
        elif trace.legendgroup == "Africa":
            trace.legendgroup = "Africa"
            trace.name = "Africa"
        
        # set initial data for line traces
        if trace.mode == 'lines':
            trace.showlegend = False
        
        # set initial data for bubble data
        else:
            if geographyLevel == "countries":
                trace.showlegend = True
            else:
                trace.showlegend = False
            # if fewer than 20 bubbles, then plot all names
            if num_bubbles <= 20:
                tracetextpositions = []
                tracetextfonts = []
                for listed_geog in trace.text:
                    #for all listed geogs
                    tracetextfonts = dict(color=color_map[trace.legendgroup])
                    if listed_geog in text_positions:
                        tracetextpositions.append(text_positions[listed_geog])
                    else:
                        tracetextpositions.append('top right')
                trace.textposition = tracetextpositions
                trace.textfont = tracetextfonts
                
            # if more than 20 bubbles then keep only the named geographies
            else:
                tracetext = []
                tracetextpositions = []
                for listed_geog in trace.text:
                    #for all listed geogs
                    tracetextfonts = dict(color=color_map[trace.legendgroup])
                    if listed_geog in text_positions:
                        tracetext.append(listed_geog)
                        tracetextpositions.append(text_positions[listed_geog])
                    else:
                        tracetext.append('')
                        tracetextpositions.append('top right')
                trace.text = tracetext
                trace.textposition = tracetextpositions
                trace.textfont = tracetextfonts
                
    
    # Update frame by frame
    for frame in fig.frames:
        #set traces in frame.data
        for trace in frame.data:
            #set frame data for all traces
            if trace.legendgroup == "Developed":
                trace.legendgroup = "Developed"
                trace.name = "Developed"
            elif trace.legendgroup == "Latin America and the Caribbean":
                trace.legendgroup = "Latin America <br>and the Caribbean"
                trace.name = "Latin America <br>and the Caribbean"
            elif trace.legendgroup == "Developing Asia and Oceania":
                trace.legendgroup = "Developing Asia <br>and Oceania"
                trace.name = "Developing Asia <br>and Oceania"
            #print(frame.name)
            
            # set frame data for line traces
            if trace.mode == 'lines':
                trace.showlegend = False
            
            # set frame data for bubble traces
            else:
                if geographyLevel == "countries":
                    trace.showlegend = True
                else:
                    trace.showlegend = False
                # if fewer than 20 bubbles, then plot all names
                if num_bubbles <= 20:
                    tracetextpositions = []
                    for listed_geog in trace.text:
                        #for all listed geogs
                        tracetextfonts = dict(color=color_map[trace.legendgroup])
                        if listed_geog in text_positions:
                            tracetextpositions.append(text_positions[listed_geog])
                        else:
                            tracetextpositions.append('top right')
                    trace.textposition = tracetextpositions
                    trace.textfont = tracetextfonts
                
                # if more than 20 bubbles keep only names geographies
                else:
                    tracetext = []
                    tracetextpositions = []
                    for listed_geog in trace.text:
                        #for all listed geogs
                        tracetextfonts = dict(color=color_map[trace.legendgroup])
                        if listed_geog in text_positions:
                            tracetext.append(listed_geog)
                            tracetextpositions.append(text_positions[listed_geog])
                        else:
                            tracetext.append('')
                            tracetextpositions.append('top right')
                    trace.text = tracetext
                    trace.textposition = tracetextpositions
                    trace.textfont = tracetextfonts

        
        #set annotations in frame, year and axes labels
        year = math.floor(float(frame.name))
        new_annotation = go.layout.Annotation(
            text=f"{year}",  # Annotation text showing the year
            showarrow=False,  # No arrow pointing to a data point
            xref="x",  # X position relative 'paper' or 'x' for 'data'
            yref="y",  # Y position relative 'paper' or 'y' for 'data'
            x= min_x + 0.15*(max_x-min_x), #   pow(10, max_x) * 0.5,  # X position (bottom right)
            y= min_y + 0.75*(max_y-min_y),  # Y position (bottom right)
            font=dict(size=45, color="grey"),  # Font size and color
            align="right"  # Align text to the right
        )
        if 'annotations' in frame.layout:
            frame.layout.annotations += (new_annotation,)
        else:
            frame.layout.annotations = (new_annotation,)
        
    
        new_annotation = go.layout.Annotation(
            text=x_var_label,  # Annotation text showing the year
            showarrow=False,  # No arrow pointing to a data point
            xref="x",  # X position relative 'paper' or 'x' for 'data'
            yref="y",  # Y position relative 'paper' or 'y' for 'data'
            x= min_x + 1*(max_x-min_x), #   pow(10, max_x) * 0.5,  # X position (bottom right)
            y= min_y + 0.07*(max_y-min_y),  # Y position (bottom right)
            font=dict(size=18, color="black"),  # Font size and color
            xanchor="right",
            align="right",
        )
        if 'annotations' in frame.layout:
            frame.layout.annotations += (new_annotation,)
        else:
            frame.layout.annotations = (new_annotation,)
            
        new_annotation = go.layout.Annotation(
            text=y_var_label,  # Annotation text showing the year
            showarrow=False,  # No arrow pointing to a data point
            xref="x",  # X position relative 'paper' or 'x' for 'data'
            yref="y",  # Y position relative 'paper' or 'y' for 'data'
            x= min_x + 0.02*(max_x-min_x), #   pow(10, max_x) * 0.5,  # X position (bottom right)
            y= min_y + 0.95*(max_y-min_y),  # Y position (bottom right)
            font=dict(size=18, color="black"),  # Font size and color
            xanchor="left",
            align="left",
        )
        if 'annotations' in frame.layout:
            frame.layout.annotations += (new_annotation,)
        else:
            frame.layout.annotations = (new_annotation,)
        
        



    #change label in slider steps

    new_steps = []
    for step in fig.layout.sliders[0].steps:
        # Check if the step label can be converted to a float and is divisible by 5
            if float(step.label) % 5 == 0:  # Check if divisible by 5
                step.label = str(round(float(step.label)))
                new_steps.append(step)  # Add step to new_steps

    # Update the slider with the new filtered steps
    fig.layout.sliders[0].steps = new_steps



  #  for step in fig.layout.sliders[0].steps:
#
#        step_value = float(step.label)
#        #print(step_value)
#       # print(str((step_value % 5)))
#        if (step_value % 5) < or (step_value % 2) > 1:  # if the value is divisible by
#            #print(step_value)
#            step.label = str(round(step_value))
#        else:
#            step.label = ''
   
       # if abs(step_value - first_year)<1:
       #     print("first year")
       #     step.label = str(int(first_year))
       # elif abs(step_value - last_year)<1:
       #     print("last year")
       #     step.label = str(int(last_year))
            
    fig.layout.sliders[0].currentvalue = dict(visible=False)
    fig.layout.sliders[0].minorticklen = 0
    fig.layout.sliders[0].ticklen = 0

    
    full_index = pd.Index(geographies, name=geography)
    if show_flags:
        progress.set(90, "adding flags...")
        
        for frame in fig.frames: #frame = fig.frames[19]
            year = frame.name
            year_data = plot_df[plot_df['year'] == float(year)]
            
            ## add in NA rows for missing data
            year_data = year_data.drop_duplicates(subset=[geography])
            year_data = year_data.set_index(geography).reindex(full_index).reset_index()

            # Create list of image annotations for this year
            image_annotations = []
            for i, row in year_data.iterrows():
                if x_log:
                    x_position = np.log10(row[x_var])
                else:
                    x_position = row[x_var]
                if y_log:
                    y_position = np.log10(row[y_var])
                else:
                    y_position = row[y_var]
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
                            'opacity': 1,
                            'layer': "above"
                            }
                          )
                else:
                    image_annotations.append(
                      {
                              'source': row['image_link'],
                              'xref': "x",
                              'yref': "y",
                              'x': x_position,
                              'y': y_position,
                              'sizex': row['normalised_size'],
                              'sizey': row['normalised_size'],
                              'xanchor': "center",
                              'yanchor': "middle",
                              'sizing': "contain",
                              'opacity': 1,
                              'layer': "above"
                              }
                          )
            
            frame.layout.images = image_annotations
            #print(frame.layout.images)
            
        now_time = time.time()
        print(f"added flags time: {now_time - start_time} seconds")
        progress.set(95, "printing plot...")
     
        
    

    ######## LABELS, AXES, AND SLIDER
    
    
    buttons = [{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 400/smoothness, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate', 'transition': {'duration': 0/smoothness}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 400/smoothness, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0/smoothness}}],
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
        'y': 0.05,
        'yanchor': 'top'
    }]
    sliders=[{
        'active': 0,
        #'currentvalue': {'prefix': 'Frame: '},
        'x': 0.1,
        'len': 0.9,
        'pad': {'b': 10},
        'ticklen': 0,
    }]
    
    
    # Adjust layout
    fig.update_layout(
        xaxis_title=None, #{'text': x_var_label},
        yaxis_title=None, #{'text': y_var_label},
      #  title={'text': f"<span style='font-size:28px; font-family:Inter;'><b>The glaring inequality of income and CO<sub>2</sub> emissions</b></span><br><span style='font-size:18px;'><i>{y_var_label} vs {x_var_label}</i></span>",
      #         'x': 0.08,
      #         'xanchor': 'left',},
        autosize=True,
        #width=1100,
        #height=700,
        updatemenus=buttons,
        #sliders=sliders,
        title_font=dict(family="Inter", size=24, color="black"),
        xaxis_title_font=dict(family="Inter", size=18, color="black"),
        yaxis_title_font=dict(family="Inter", size=18, color="black"),
        font=dict(family="Inter", size=14, color="black"),
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
    
    animation_opts = {'frame': {'duration': 400/smoothness, 'redraw': True},'transition': {'duration': 0/smoothness}}
    #fig.write_html('/Users/edbaker/UN_projects/c02emmisions/plotly_animation.html', full_html=True, auto_play=True, default_width='95vw', default_height='95vh', div_id='id_plot-container', animation_opts=animation_opts)
    
    # Path to your HTML file
    #html_file = '/Users/edbaker/UN_projects/c02emmisions/plotly_animation.html'
    
    # Read the HTML file
 #   with open(html_file, 'r') as f:
 #       raw = f.read()
    
    # We will look for a rect element with the class "bg" and modify the width attribute
  #  import re
  #  
    # Regex pattern to find <rect> with class "bg"
    # This pattern captures the rect element and the current width attribute (if present)
 #   pattern = r'(<rect[^>]*class="bg"[^>]*)(width="[^"]*")?([^>]*>)'
    
    # Replacement pattern to inject the desired width (e.g., 200) directly into the rect element
  #  new_width = r'\1 width="250" \3'
    
    # Perform the replacement in the raw HTML
  #  fixed = re.sub(pattern, new_width, raw)
    
    # Write the modified content back to the file
  #  with open(html_file, 'w') as f:
  #      f.write(fixed)

    
    now_time = time.time()
    print(f"plot saved time: {now_time - start_time} seconds")
    
    
    #print(figScatter.layout)
    #print(fig.data)
    # print(fig.layout)
    
    return(fig)
    
#fig = createCountryBubbleGraph()    
# Save the figure as HTML
#fig.write_html('/Users/edbaker/UN_projects/c02emmisions/plotly_animation.html')
