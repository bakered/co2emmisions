#. %reset -f

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
#from plotly.subplots import make_subplots
#import copy
import time

def createCountryBubbleGraph(geographyLevel='countries', 
                             x_var = 'gdp_per_capita', 
                             y_var = 'co2_per_capita', 
                             size_var = 'co2',
                             race_var = 'accumulated_co2',
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
    
    color_map = {'LDCs': '#FFC800', 'Developed': '#009EDB', 'Developing': '#ED1847'}
    labels = {"co2_per_capita": "CO2 per capita",
             "gdp_per_capita": "GDP per capita",
             "co2": "Yearly CO2 (tons)",
             "accumulated_co2": "Accumulated CO2", 
             "pop": "Population", 
             "gdp": "GDP (millions)", 
              }
    
    now_time = time.time()
    print(f"start function time: {now_time - start_time} seconds")
    progress.set(2, "Loading data...")
    
    # Import the data
    #data = pd.read_csv("dataCountries.csv")
    #data = pd.read_csv(open_url('https://github.com/bakered/co2emmisions/blob/main/src_shiny_app/dataPlot1.csv'))
    if geographyLevel == "countries":
        infile = Path(__file__).parent / "dataCountries.csv"
    else:
        infile = Path(__file__).parent / "dataRegions.csv"
    data = pd.read_csv(infile)
    
    now_time = time.time()
    print(f"loaded data time: {now_time - start_time} seconds")
    progress.set(3, "creating bubbles...")
    
    if geographyLevel == 'countries': 
        data['year'] = data['year'].astype(int)
        # create image link from ISO2
        data['ISO2']= data['ISO2'].astype(str)
        data['image_link'] = data['ISO2'].apply(lambda iso: f"https://hatscripts.github.io/circle-flags/flags/{iso.lower()}.svg")
        data['geography'] = data['ISO3']
    else:
        data['geography'] = data['region2']

    
    # Filter the DataFrame based on the list of ISO3 codes
    plot_df = data[(data['geography'].isin(geography_list)) & (data['year'] > start_year)].copy()

    #geography_list=['ARG', 'AUS', 'BRA', 'CAN', 'CHN', 'FRA', 'DEU', 'IND', 'IDN', 'ITA', 'JPN', 'MEX', 'RUS', 'SAU', 'ZAF', 'KOR', 'TUR', 'GBR', 'USA', 'SGP', 'PNG', 'MYS', 'BRN', 'IRN', 'OMN']
    
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
    
    # Calculate weighted 95% percentile
    index_max_x = plot_df.groupby('geography')[x_var].idxmax().dropna()
    index_max_y = plot_df.groupby('geography')[y_var].idxmax().dropna()
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
    plot_df['bubble_size'] = plot_df[size_var] + bubble_size
    plot_df['bubble_size'] = plot_df['bubble_size'].fillna(0)
    scatter_size_max_parameter = 25

    print(plot_df['bubble_size'])
    
    if max_x>max_y:
        max_bubble_size_wanted = (max_x/10)*flag_size
    else:
        max_bubble_size_wanted = (max_y/5)*flag_size
    
    
    # Add normalized_size column such that the value is the diameter making 
    plot_df[size_var] = plot_df[size_var].fillna(0)
    plot_df.loc[:, 'normalized_size'] = 2 * np.sqrt(plot_df[size_var] /3.141592653589793) 
    co2_max = plot_df['normalized_size'].max()
    # scale column such that the co2_max will be the size of the bubble you want measured in x-axis units. 
    # (warning: i think only works if x-axis is larger than y-axis)
    plot_df.loc[:, 'normalized_size'] *= (max_bubble_size_wanted / co2_max)
    
    #this is used later in flag addition
    geographies = plot_df['geography'].unique()
    
    if geographyLevel == "countries":
        # Create the base plot with Plotly Express
        figScatter = px.scatter(
            plot_df, 
            x=x_var, 
            y=y_var, 
            color='region2',
            color_discrete_map=color_map,
            size='bubble_size', 
            hover_name='geography',
            animation_frame='year', 
            size_max=scatter_size_max_parameter,
            template="plotly_white"
        )
        figScatter.update_layout(
            legend_title_font=dict(size=15, family="Times New Roman"),
            legend=dict(
                title='Region',
                x=0.15,            # x-coordinate of the legend (0 = left, 1 = right)
                y=0.85,            # y-coordinate of the legend (0 = bottom, 1 = top)
                xanchor='left', # Positioning relative to the x-coordinate (left)
                yanchor='top',  # Positioning relative to the y-coordinate (top)
            )
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
            color='geography',
            color_discrete_map=color_map,
            size='bubble_size', 
            text='geography',
            hover_name='geography',
            animation_frame='year', 
            size_max=scatter_size_max_parameter,
            template="plotly_white"
        )
        figScatter.update_layout(
            legend_title_font=dict(size=15, family="Times New Roman"),
            legend=dict(
                x=0.15,            # x-coordinate of the legend (0 = left, 1 = right)
                y=0.85,            # y-coordinate of the legend (0 = bottom, 1 = top)
                xanchor='left', # Positioning relative to the x-coordinate (left)
                yanchor='top',  # Positioning relative to the y-coordinate (top)
            )
        )
        figScatter.update_traces(textposition='top right')
        figScatter.update_traces(marker=dict(opacity=0.8))
        text_positions = {
            'LDCs': 'top right',
            'Developing': 'top center',
            'Developed': 'bottom center'
        }
        # Update traces with positions and labels
        for trace in figScatter.data:
            if trace.name in text_positions:
                trace.textposition = text_positions[trace.name]
                
        
    now_time = time.time()
    print(f"Created scatter time: {now_time - start_time} seconds")
    progress.set(10, "calculating lines data...")
    
    # Extract color mapping from figLine
    #color_map = {trace.name: trace.marker.color for trace in figScatter.data if 'color' in trace.marker}
    
    # add log axes if necessary
    if x_log:
        figScatter.update_xaxes(type="log")
    
    if y_log:
        figScatter.update_yaxes(type='log')
        
    # following code does not work with shiny? but with html
    figScatter.update_traces(
        customdata=plot_df[['geography', 'gdp']].values,
        hovertemplate="<br>".join([
            "<strong>%{customdata[0]}</strong><br>",
            "GDP (millions): %{customdata[1]:,.0f}",
            "<extra></extra>"  # Removes the trace name from hover
    ])
    )
    
    
    if leave_trace:
        # xxxxx add a line of best fit : smooth curve..or pre-calculate and save in data?
        #  Loop over each year in the DataFrame
        expanded_rows = pd.DataFrame()
        for year in plot_df['year'].unique(): # year=plot_df['year'].unique()[12]
            # Filter the rows for the current year and all previous years
            filtered_rows = plot_df[plot_df['year'].astype(int) <= int(year)].copy()
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
            color='region2',
            line_group='geography',
            color_discrete_map=color_map,
            #line_group='geography',  # Group by geography to draw a separate line for each
            #hover_name='geography', 
            animation_frame='year',
            template="plotly_white"
            )
        figLine.update_traces(hoverinfo='skip', hovertemplate=None)
        figLine.update_layout(showlegend=False)
        figLine.update_traces(opacity=0.8)
        
        now_time = time.time()
        print(f"created line graph time: {now_time - start_time} seconds")
        progress.set(70, "Combining plots...")
        
        # add log axes
        if x_log:
            figLine.update_xaxes(type="log")
        if y_log:
            figLine.update_yaxes(type='log')
          
        #fig.update_layout(
        #  xaxis=dict(range=[0, plot_df[x_var].max() * 1.1]),  # Adjust range as needed
        #  yaxis=dict(range=[0, plot_df[y_var].max() * 1.1]),  # Adjust range as needed
        #)
    
        # Hide legend for figLine traces
        for trace in figLine.data:
            trace.showlegend = False
            
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
    
    
    # Add flags to each frame
    full_index = pd.Index(geographies, name='geography')
    if geographyLevel == "countries" and not x_log and not y_log and show_flags:
        progress.set(90, "adding flags...")
        for frame in fig.frames: #frame = fig.frames[19]
            #print(frame.layout.images)
            year = frame.name
            #print(year)
            # Filter data for the specific year
            #if leave_trace:
            #  year_data = plot_df[plot_df['year'] <= int(year)]
            #  ## add in NA rows for missing data
            #  full_years = list(range(1970, int(year)+1))
            #  full_index = pd.MultiIndex.from_product([countries, full_years], names=['geography', 'year'])
            #  year_data = year_data.set_index(['geography', 'year']).reindex(full_index).reset_index()
            #else:
            year_data = plot_df[plot_df['year'] == int(year)]
            ## add in NA rows for missing data
     
            
            year_data = year_data.set_index('geography').reindex(full_index).reset_index()
      
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
            # Update the layout of the frame to include the images
            frame.layout.images = image_annotations
            #print(frame.layout.images)
            x_max_frame = year_data[x_var].max() + max_bubble_size_wanted
            y_max_frame = year_data[y_var].max()  * 1.1
            custom_xaxis = {'range': [0, x_max_frame]}
            custom_yaxis = {'range': [0, y_max_frame]}
            # Assign the custom axes to the frame's layout
            # hashed out because makes plot jittery
            # frame.layout.xaxis = custom_xaxis
            # frame.layout.yaxis = custom_yaxis
        now_time = time.time()
        print(f"added flags time: {now_time - start_time} seconds")
        progress.set(95, "printing plot...")
     
        
    #### add year label on each frame
    # Define the position for the annotation (bottom right corner)
    annotation_x = max_x * 0.95  # X position (relative to the plot)
    annotation_y = max_y * 0.05  # Y position (relative to the plot)
    new_frames = []
    # Loop over each frame and add a text annotation
    for frame in fig.frames:
        # Create a deep copy of the frame to modify
        new_frame = go.Frame(data=frame.data, name=frame.name, layout=frame.layout)
        # Add annotation for the current year
        # Create the new annotation
        new_annotation = go.layout.Annotation(
            text=f"{frame.name}",  # Annotation text showing the year
            showarrow=False,  # No arrow pointing to a data point
            xref="x",  # X position relative 'paper' or 'x' for 'data'
            yref="y",  # Y position relative 'paper' or 'y' for 'data'
            x=annotation_x,  # X position (bottom right)
            y=annotation_y,  # Y position (bottom right)
            font=dict(size=45, color="grey"),  # Font size and color
            align="right"  # Align text to the right
        )
        
        # Check if the frame already has annotations and append the new one
        if 'annotations' in new_frame.layout:
            new_frame.layout.annotations += (new_annotation,)
        else:
            new_frame.layout.annotations = (new_annotation,)
        # Append the frame with the annotation to the new_frames list
        new_frames.append(new_frame)
    fig.frames = new_frames


    y_var_label = labels.get(y_var, y_var)  # Default to y_var if not found in labels
    x_var_label = labels.get(x_var, x_var)  # Default to x_var if not found in labels
    # Adjust layout for better appearance
    fig.update_layout(
        xaxis_title=x_var_label,
        yaxis_title=y_var_label,
       # title=f"<h1>The glaring inequality of income and CO2 emissions</h1> <br> <h3>{y_var} vs {x_var}</h3>",
        title={'text': f"<span style='font-size:28px;'><b>The glaring inequality of income and CO2 emissions</b></span><br><span style='font-size:18px;'><i>{y_var_label} vs {x_var_label}</i></span>",
               'x': 0.08,
               'xanchor': 'left',},
        autosize=True,
        #width=1100,
        #height=700,
    )
    
    if fixed_axes:
      fig.update_layout(
        xaxis=dict(range=[min_x, max_x]),
        yaxis=dict(range=[min_y, max_y])
        )


    
        
    
    if True:
        # Update layout 
        fig.update_layout(
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 250, 'redraw': False}, 'fromcurrent': True, 'mode': 'immediate', 'transition': {'duration': 250}}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 250, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 250}}],
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
            }],
        )
  
    
    fig.update_layout(
        title_font=dict(family="Times New Roman", size=24, color="black"),
        xaxis_title_font=dict(family="Times New Roman", size=18, color="black"),
        yaxis_title_font=dict(family="Times New Roman", size=18, color="black"),
        font=dict(family="Times New Roman", size=14, color="black")
    )
    
    now_time = time.time()
    print(f"plot created time: {now_time - start_time} seconds")
    
   # fig.write_html('/Users/edbaker/UN_projects/c02emmisions/plotly_animation.html')
    
    now_time = time.time()
    print(f"plot saved time: {now_time - start_time} seconds")
    
    return(fig)
    
#fig = createCountryBubbleGraph()    
# Save the figure as HTML
#fig.write_html('/Users/edbaker/UN_projects/c02emmisions/plotly_animation.html')
