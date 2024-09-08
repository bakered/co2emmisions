#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 06:59:49 2024

@author: edbaker
"""


from shiny import ui, render, App
import numpy as np
from PlotlyPlot1 import createCountryBubbleGraph  # Import the function from the other file

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

img_urls = [f'https://cdn.rawgit.com/lipis/flag-icon-css/master/flags/4x3/{country}.svg' for country in countries]


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select(id="x_var", label="X Variable:", 
                        choices={"co2": "CO2", "population": "Population", "gdp": "GDP", 
                                 "gdp_per_capita": "GDP per Capita", "co2_per_capita": "CO2 per Capita"},
                        selected="gdp_per_capita"),
        
        ui.input_select(id="y_var", label="Y Variable:", 
                        choices={"co2": "CO2", "population": "Population", "gdp": "GDP", 
                                 "gdp_per_capita": "GDP per Capita", "co2_per_capita": "CO2 per Capita"},
                        selected="co2_per_capita"),
        
        ui.input_select(id="size_var", label="Size Variable:", 
                        choices={"co2": "CO2", "population": "Population", "gdp": "GDP", 
                                 "gdp_per_capita": "GDP per Capita", "co2_per_capita": "CO2 per Capita"},
                        selected="co2"),
        
        ui.input_selectize(id="geography_list", label="Geography List:", 
                           choices=countries2,
                           multiple=True, 
                           selected=["ARG", "AUS", "BRA", "CAN", "CHN", "FRA", "DEU", "IND", "IDN", 
                                     "ITA", "JPN", "MEX", "RUS", "SAU", "ZAF", "KOR", "TUR", "GBR", "USA"]),
        ui.input_numeric(id="bubble_size", label="Size bubble: (x times bigger)", min=0.01, max=100, value=1),
        ui.input_checkbox(id="fixed_axes", label="Fixed Axes", value=True),
        ui.input_checkbox(id="leave_trace", label="Leave Trace", value=True),
        
    ),
    ui.tags.style("""
        .plot-container {
            height: 700px;
        }
    """),
    ui.output_ui("plot"),
    #ui.output_ui("plot", class="plot-container"),
    #ui.output_plot("plot"),
    )

def server(input, output, session):
    @output
    #@render.plot
    #@render_widget
    @render.ui
    def plot():
        plot = createCountryBubbleGraph(
            x_var=input.x_var(),
            y_var=input.y_var(),
            size_var=input.size_var(),
            geography_list=input.geography_list(),
            bubble_size=input.bubble_size(),
            fixed_axes=input.fixed_axes(),
            leave_trace=input.leave_trace()
        )
        return ui.HTML(plot.to_html(full_html=False))


app = App(ui=app_ui, server=server) 