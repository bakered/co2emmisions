#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 06:59:49 2024

@author: edbaker
"""


from shiny import ui, render, App
import numpy as np
import matplotlib.pyplot as plt
from PlotlyPlot1 import createCountryBubbleGraph  # Import the function from the other file
from shinywidgets import output_widget, render_widget  


app_ui = ui.page_fluid(
    ui.input_numeric(id="max_bubble_size_wanted", label="size bubble:", min=5, max=3000000, value=6000),
    #output_widget("plot"), 
    ui.output_ui("plot"),
)

#app_ui = ui.page_fluid(
#    ui.layout_sidebar(
#        ui.sidebar(
#            ui.h2("Page Charts"),
#            ui.hr(),
#            ui.input_slider(id="slider", label="size bubble:", min=5, max=300000, value=100000),
#            #ui.output_plot(id="histogram")
#        ),
#        ui.output_plot(id="plot")
#    )
#)

def server(input, output, session):
    @output
    #@render.plot
    #@render_widget
    @render.ui
    def plot():
        print("input" + str(input.max_bubble_size_wanted()))
        plot = createCountryBubbleGraph(max_bubble_size_wanted=input.max_bubble_size_wanted(), fixed_axes = True)  # 
        return ui.HTML(plot.to_html())



app = App(ui=app_ui, server=server) 