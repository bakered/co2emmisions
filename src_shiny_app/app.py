#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 06:59:49 2024

@author: edbaker
"""


from shiny import ui, render, App, reactive
from PlotlyPlot1 import createCountryBubbleGraph  # Import the function from the other file
import time
import asyncio

start_time = time.time()


#from shinywidgets import output_widget, render_widget  

countries = [
    "un", "cn", "sg", "my", "bn", "jp", "au", "us", "ca", "ir", "om",
    "bd", "pg", "vn", "kr", "tw", "hk", "id", "za", "sn", "ph", "ru",
    "eg", "il", "tr", "ro", "it", "mc", "gr", "bg", "in", "lk", "mt",
    "sa", "nl", "be", "th", "fr", "de", "kh", "mm", "kw", "iq", "ki",
    "mh", "mx", "ae", "bj", "gh", "dj", "mz", "sc", "gn", "sb", "gy",
    "bb", "lc", "es", "ch", "ao", "ye", "ge", "ua", "mu", "gw", "as",
    "pl", "is", "no", "ma", "tg", "ng", "se", "dk", "gb", "ga", "so",
    "cm", "tt", "bs", "hr", "fi", "pa", "ec", "ar", "ve", "vc", "sr",
    "gd", "kn", "tc", "ag", "ht", "me", "at", "hu", "sk", "st", "vg",
    "do", "bz", "pt", "hn", "co", "cr", "br", "ni", "al", "ee", "sd",
    "lb", "jo", "ly", "cy", "qa", "cl", "lv", "lt", "sy", "md", "ci",
    "tn", "dz", "fo", "gl", "uy", "ie", "km", "tz", "lr", "cv", "bh",
    "tm", "az", "kz", "si", "gi", "py", "ky", "cu", "jm", "kp", "mr",
    "gq", "ke", "rs", "pk", "ms", "mv", "gu", "cg", "gm", "fk", "eh",
    "sl", "mg", "aw", "pe", "pf", "nz", "dm", "bm", "er", "sv", "cd",
    "mo", "ws", "fj", "to", "fm", "tl", "nc", "tv", "gt", "vu", "nr",
    "mp", "pw", "nu", "ck", "sh", "wf"
]
countries = [
    "AW", "AF", "AO", "AI", "AL", "AD", "AE", "AR", "AM", "AG", 
    "AU", "AT", "AZ", "BI", "BE", "BJ", "BQ", "BF", "BD", "BG", "BH", 
    "BS", "BA", "BY", "BZ", "BM", "BO", "BR", "BB", "BN", "BT", "BW", 
    "CF", "CA", "CH", "CL", "CN", "CI", "CM", "CD", "CG", "CK", "CO", 
    "KM", "CV", "CR", "CU", "CW", "CY", "CZ", "DE", "DJ", "DM", "DK", 
    "DO", "DZ", "EC", "EG", "ER", "ES", "EE", "ET", "FI", "FJ", "FR", 
    "FO", "FM", "GA", "GB", "GE", "GH", "GN", "GM", "GW", "GQ", "GR", 
    "GD", "GL", "GT", "GY", "HK", "HN", "HR", "HT", "HU", "ID", "IN", 
    "IE", "IR", "IQ", "IS", "IL", "IT", "JM", "JO", "JP", "KZ", "KE", 
    "KG", "KH", "KI", "KR", "KW", "LA", "LB", "LR", "LY", "LC", "LI", 
    "LK", "LS", "LT", "LU", "LV", "MO", "MA", "MD", "MG", "MV", "MX", 
    "MH", "MK", "ML", "MT", "MM", "ME", "MN", "MZ", "MR", "MS", "MU", 
    "MW", "MY", "NA", "NC", "NE", "NG", "NI", "NU", "NL", "NO", "NP", 
    "NR", "NZ", "OM", "PK", "PA", "PE", "PH", "PW", "PG", "PL", "KP", 
    "PT", "PY", "PS", "PF", "QA", "RO", "RU", "RW", "SA", "SD", "SN", 
    "SG", "SH", "SB", "SL", "SV", "SO", "PM", "RS", "SS", "ST", "SR", 
    "SK", "SI", "SE", "SZ", "SX", "SC", "SY", "TC", "TD", "TG", "TH", 
    "TJ", "TM", "TL", "TO", "TT", "TN", "TR", "TV", "TW", "TZ", "UG", 
    "UA", "UY", "US", "UZ", "VC", "VE", "VG", "VN", "VU", "WF", "WS", 
    "YE", "ZA", "ZM", "ZW"
    ]

countries2 = [
    "CHN", "SGP", "MYS", "BRN", "JPN", "AUS", "USA", "CAN", "IRN",
    "OMN", "BGD", "PNG", "VNM", "KOR", "TWN", "HKG", "IDN", "ZAF", "SEN",
    "PHL", "RUS", "EGY", "ISR", "TUR", "ROU", "ITA", "MCO", "GRC", "BGR",
    "IND", "LKA", "MLT", "SAU", "NLD", "BEL", "THA", "FRA", "DEU", "KHM",
    "MMR", "KWT", "IRQ", "KIR", "MHL", "MEX", "ARE", "BEN", "GHA", "DJI",
    "MOZ", "SYC", "GIN", "SLB", "GUY", "BRB", "LCA", "ESP", "CHE", "AGO",
    "YEM", "GEO", "UKR", "MUS", "GNB", "ASM", "POL", "ISL", "NOR", "MAR",
    "TGO", "NGA", "SWE", "DNK", "GBR", "GAB", "SOM", "CMR", "TTO", "BHS",
    "HRV", "FIN", "PAN", "ECU", "ARG", "VEN", "VCT", "SUR", "GRD", "KNA",
    "TCA", "ATG", "HTI", "MNE", "AUT", "HUN", "SVK", "STP", "VGB", "DOM",
    "BLZ", "PRT", "HND", "COL", "CRI", "BRA", "NIC", "ALB", "EST", "SDN",
    "LBN", "JOR", "LBY", "CYC", "QAT", "CHL", "LVA", "LTU", "SYR", "MDA",
    "CIV", "TUN", "DZA", "FRO", "GRL", "URY", "IRL", "COM", "TZA", "LBR",
    "CPV", "BHR", "TKM", "AZE", "KAZ", "SVN", "GIB", "PRY", "CYM", "CUB",
    "JAM", "PRK", "MRT", "GNQ", "KEN", "SRB", "PAK", "MSR", "MDV", "GUM",
    "COG", "GMB", "FLK", "ESH", "SLE", "MDG", "ABW", "PER", "PYF", "NZL",
    "DMA", "BMU", "ERI", "SLV", "COD", "MAC", "WSM", "FJI", "TON", "FSM",
    "TLS", "NCL", "TUV", "GTM", "VUT", "NRU", "MNP", "PLW", "NIU", "COK",
    "SHN", "WLF"
]


countries2 = ["ABW", "AFG", "AGO", "AIA", "ALB", "AND", "ARE", "ARG", "ARM", 
              "ATG", "AUS", "AUT", "AZE", "BDI", "BEL", "BEN", "BES", "BFA", 
              "BGD", "BGR", "BHR", "BHS", "BIH", "BLR", "BLZ", "BMU", "BOL", 
              "BRA", "BRB", "BRN", "BTN", "BWA", "CAF", "CAN", "CHE", "CHL", 
              "CHN", "CIV", "CMR", "COD", "COG", "COK", "COL", "COM", "CPV", 
              "CRI", "CUB", "CUW", "CYP", "CZE", "DEU", "DJI", "DMA", "DNK", 
              "DOM", "DZA", "ECU", "EGY", "ERI", "ESP", "EST", "ETH", "FIN", 
              "FJI", "FRA", "FRO", "FSM", "GAB", "GBR", "GEO", "GHA", "GIN", 
              "GMB", "GNB", "GNQ", "GRC", "GRD", "GRL", "GTM", "GUY", "HKG", 
              "HND", "HRV", "HTI", "HUN", "IDN", "IND", "IRL", "IRN", "IRQ", 
              "ISL", "ISR", "ITA", "JAM", "JOR", "JPN", "KAZ", "KEN", "KGZ", 
              "KHM", "KIR", "KOR", "KWT", "LAO", "LBN", "LBR", "LBY", "LCA", 
              "LIE", "LKA", "LSO", "LTU", "LUX", "LVA", "MAC", "MAR", "MDA", 
              "MDG", "MDV", "MEX", "MHL", "MKD", "MLI", "MLT", "MMR", "MNE", 
              "MNG", "MOZ", "MRT", "MSR", "MUS", "MWI", "MYS", "NAM", "NCL", 
              "NER", "NGA", "NIC", "NIU", "NLD", "NOR", "NPL", "NRU", "NZL", 
              "OMN", "PAK", "PAN", "PER", "PHL", "PLW", "PNG", "POL", "PRK", 
              "PRT", "PRY", "PSE", "PYF", "QAT", "ROU", "RUS", "RWA", "SAU", 
              "SDN", "SEN", "SGP", "SHN", "SLB", "SLE", "SLV", "SOM", "SPM", 
              "SRB", "SSD", "STP", "SUR", "SVK", "SVN", "SWE", "SWZ", "SXM", 
              "SYC", "SYR", "TCA", "TCD", "TGO", "THA", "TJK", "TKM", "TLS", 
              "TON", "TTO", "TUN", "TUR", "TUV", "TWN", "TZA", "UGA", "UKR", 
              "URY", "USA", "UZB", "VCT", "VEN", "VGB", "VNM", "VUT", "WLF", 
              "WSM", "YEM", "ZAF", "ZMB", "ZWE"]

g20_countries = ["ARG", "AUS", "BRA", "CAN", "CHN", "FRA", "DEU", "IND", "IDN", 
          "ITA", "JPN", "MEX", "RUS", "SAU", "ZAF", "KOR", "TUR", "GBR", "USA"]

img_urls = [f'https://cdn.rawgit.com/lipis/flag-icon-css/master/flags/4x3/{country}.svg' for country in countries]

#regions = ["LDCs", "Developed", "Developing"] 
region1s = ["Developed", "Developing Asia and Oceania", "Latin America and the Carribean", "Africa"]


labels = {"co2_per_capita": "CO2 per capita (Tons)",
         "gdp_per_capita": "GDP per capita (2011 Dollars, PPP)",
         "co2": "Yearly CO2 (Kilotons)",
         "accumulated_co2": "Accumulated CO2 (Kilotons)", 
         "pop": "Population", 
         "gdp": "GDP (Millions)", 
          }

gochi_font_css = """
<style type="text/css">
    @font-face {
        font-family: 'Helvetica Neue LT Std 45 Light';
        src: url('https://github.com/esambino/H_and_L/blob/master/font/helvetica-neue-lt-std-45-light.otf') format("opentype"); 
    }
    body {
        font-family: "HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif; 
    }
</style>
"""




app_ui = ui.page_fluid(
    ui.HTML(gochi_font_css),
    ui.tags.style("""
                  
        .js-plotly-plot,
        .plot-container,
        .svg-container {
            height: 85vh;
            }
        
        .sidebar {
            width: 600px;  
        }
        
        .custom-green-button {
            background-color: #72BF44;
            color: white;
            }
        
        .body, button, input, select, textarea {
            color: #72BF44;
            }
        
        aside .btn,
        aside p,
        aside input,
        aside textarea,
        aside label,
        aside select {
            font-family: "HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;  
        }
        
    """),
     ui.layout_sidebar(
            ui.sidebar(
                # Add a button
                ui.tags.p("Click Plot button for changes to take effect:"),
                ui.input_action_button(id="plot_button", label="Plot", class_="custom-green-button"),
                ui.input_select(id="datasource", label="Data source:", 
                                choices={"GCP and Maddison":"GCP and Maddison", "World Bank WDI":"World Bank WDI"},
                                selected="GCP and Maddison"),
                ui.input_select(id="geographyLevel", label="Geography:", 
                                choices={"countries": "Countries", "regions": "Regions"},
                                selected="countries"),
                
                ui.input_select(id="x_var", label="X Variable:", 
                                choices=labels,
                                selected="gdp_per_capita"),
                
                ui.input_select(id="y_var", label="Y Variable:", 
                                choices=labels,
                                selected="co2_per_capita"),
                
                ui.input_select(id="size_var", label="Size Variable:", 
                                choices=labels,
                                selected="co2"),
                
                ui.input_numeric(id="smoothness", label="Smoothness:", min=1, max=50, value=5),
                ui.input_numeric(id="start_year", label="Start Year:", min=1820, max=2021, value=1951),
                ui.input_action_button(id="select_all_countries", label="Select all countries"),
                ui.input_action_button(id="select_g20", label="Select G20"),
                ui.input_selectize(id="geography_list", label="Show country:", 
                                   choices=region1s+countries2,
                                   multiple=True, 
                                   selected=countries2
                                   ),
                
                ui.input_numeric(id="bubble_similarity", label="Bubble similar size (increase to make bubbles more similar): ", min=0, value=1000000),
                ui.input_numeric(id="flag_size", label="Flag size: (x times bigger)", min=0.01, max=100, value=1),
                ui.input_numeric(id="bubble_size", label="Bubble size: (x times bigger)", min=0.01, max=100, value=1),
                ui.input_numeric(id="rolling_mean_years", label="Rolling mean for trace line", min=1, max=100, value=1),
                
                ui.input_checkbox(id="fixed_axes", label="Fixed Axes", value=True),
                ui.input_checkbox(id="leave_trace", label="Leave Trace", value=False),
                ui.input_checkbox(id="x_log", label="x axis log", value=True),
                ui.input_checkbox(id="y_log", label="y axis log", value=False),
                ui.input_checkbox(id="show_flags", label="Show Flags", value=True),
                ui.input_checkbox(id="use_loess", label="Use loess for lines", value=True),
                
                
                bg="#f8f8f8", open="closed",
                ),
            ui.tags.style("""
                          .sidebar {
                              width: 600px;  /* Adjust width as needed */
                              }
                          """),
            ui.card(
                #output_widget("plot")
                ui.output_ui("plot", fill=True)
                ),
            )
    )

def server(input, output, session):  
    
    # does this always react--- could make more efficient?    
    @reactive.effect
    @reactive.event(input.geographyLevel)
    def _():
        if input.geographyLevel() == "countries":
            ui.update_selectize("geography_list", choices=region1s+countries2, selected=countries2, label="Show country:")
            ui.update_numeric("bubble_similarity", value =1000000)
            ui.update_checkbox('leave_trace', value=False)
        else:
            ui.update_selectize("geography_list", choices=region1s, selected=region1s, label="Show region:")
            ui.update_numeric("bubble_similarity", value =1000000)
            ui.update_checkbox('leave_trace', value=True)
       
    @reactive.Effect
    @reactive.event(input.select_all_countries)
    def select_all_countries():
        ui.update_select("geographyLevel", selected="countries")
        ui.update_selectize("geography_list", choices=region1s+countries2, selected=countries2, label="Show country:")
        ui.update_numeric("bubble_similarity", value =1000000)
        ui.update_checkbox('leave_trace', value=False)

    # Handle "Select G20" button click
    @reactive.Effect
    @reactive.event(input.select_g20)
    def select_g20_countries():
        ui.update_select("geographyLevel", selected="countries")
        ui.update_selectize("geography_list", choices=region1s+countries2, selected=g20_countries, label="Show country:")
        ui.update_numeric("bubble_similarity", value =1000000)
        ui.update_checkbox('leave_trace', value=False)
        
        
    #@output
    #@render.plot
    #@render_widget
    #@render.ui
    #def out_plot():
    #   return ui.HTML("<p>Click the button to generate the plot.</p>")  #
    
    #@reactive.effect
    @render.ui
    @reactive.event(input.plot_button, ignore_none=False)  # React to button click
    async def plot():
        with ui.Progress(max=100) as progress:
            plot = createCountryBubbleGraph(
                datasource=input.datasource(),
                geographyLevel=input.geographyLevel(),
                x_var=input.x_var(),
                y_var=input.y_var(),
                size_var=input.size_var(),
                smoothness=input.smoothness(),
                rolling_mean_years=input.rolling_mean_years(),
                start_year=input.start_year(),
                geography_list=input.geography_list(),
                bubble_similarity=input.bubble_similarity(),
                bubble_size=input.bubble_size(),
                flag_size=input.flag_size(),
                fixed_axes=input.fixed_axes(),
                leave_trace=input.leave_trace(),
                x_log=input.x_log(),
                y_log=input.y_log(),
                show_flags=input.show_flags(),
                use_loess=input.use_loess(),
                start_time=start_time,
                progress=progress,
            )
            # Generate the HTML string
            animation_opts = {'frame': {'duration': 400/input.smoothness(), 'redraw': True},'transition': {'duration': 0/input.smoothness()}}
            html_content = plot.to_html(full_html=True, auto_play=True, default_width='90vw', default_height='90vh', div_id='id_plot-container', animation_opts=animation_opts)
            
            #print(type(html_content))
            # Replace "Times New Roman" with "Helvetica Neue LT Std 45 Light"
            #html_content = html_content.replace('Times New Roman', 'Helvetica')
            # Save the modified HTML to a file
            #with open('/Users/edbaker/UN_projects/c02emmisions/html_content.html', 'w') as f:
            #    f.write(html_content)

        return ui.HTML(html_content)
        #output.out_plot.set_render(ui.HTML(plot.to_html(full_html=True)))
        
     
    


app = App(ui=app_ui, server=server) 