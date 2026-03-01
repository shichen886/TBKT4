import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ChartConfig:
    """Knowledge Tracking System - Chart Configuration
    Based on Educational App color palette and modern design principles
    """
    
    PRIMARY = "#4F46E5"
    SECONDARY = "#818CF8"
    CTA = "#F97316"
    SUCCESS = "#10B981"
    WARNING = "#F59E0B"
    ERROR = "#EF4444"
    INFO = "#3B82F6"
    
    BACKGROUND = "#EEF2FF"
    SURFACE = "#FFFFFF"
    TEXT_PRIMARY = "#1E1B4B"
    TEXT_SECONDARY = "#4B5563"
    BORDER = "#C7D2FE"
    
    COLOR_SCALES = {
        'mastery': ['#EF4444', '#F59E0B', '#10B981'],
        'accuracy': ['#EF4444', '#F59E0B', '#FBBF24', '#10B981'],
        'progress': ['#E0E7FF', '#C7D2FE', '#A5B4FC', '#818CF8', '#6366F1', '#4F46E5'],
        'skills': ['#4F46E5', '#818CF8', '#A78BFA', '#C084FC', '#E879F9', '#F472B6', '#FB7185', '#F87171']
    }
    
    @staticmethod
    def get_layout_template():
        """Get unified layout template for all charts"""
        return go.layout.Template(
            layout=go.Layout(
                font=dict(
                    family="-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
                    size=14,
                    color=ChartConfig.TEXT_PRIMARY
                ),
                paper_bgcolor='rgba(255, 255, 255, 0.8)',
                plot_bgcolor='rgba(255, 255, 255, 0.5)',
                hoverlabel=dict(
                    bgcolor="rgba(255, 255, 255, 0.95)",
                    bordercolor=ChartConfig.PRIMARY,
                    font_size=14,
                    font_color=ChartConfig.TEXT_PRIMARY
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor=ChartConfig.BORDER,
                    borderwidth=1
                )
            )
        )
    
    @staticmethod
    def create_bar_chart(data, x_col, y_col, title, color_col=None, **kwargs):
        """Create styled bar chart for knowledge mastery visualization"""
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=title,
            color=color_col,
            color_continuous_scale=ChartConfig.COLOR_SCALES['mastery'],
            **kwargs
        )
        
        fig.update_traces(
            marker_line_color=ChartConfig.PRIMARY,
            marker_line_width=1,
            hovertemplate='<b>%{x}</b><br>%{y:.2%}<extra></extra>'
        )
        
        fig.update_layout(
            template=ChartConfig.get_layout_template(),
            xaxis_title=kwargs.get('labels', {}).get(x_col, x_col),
            yaxis_title=kwargs.get('labels', {}).get(y_col, y_col),
            yaxis_tickformat='.0%',
            showlegend=False if color_col is None else True
        )
        
        return fig
    
    @staticmethod
    def create_line_chart(data, y_col, title, x_col=None, **kwargs):
        """Create styled line chart for learning trends"""
        if x_col is None:
            data = data.reset_index()
            x_col = 'index'
        
        fig = px.line(
            data,
            x=x_col,
            y=y_col,
            title=title,
            **kwargs
        )
        
        fig.update_traces(
            line=dict(color=ChartConfig.PRIMARY, width=3),
            fill='tozeroy',
            fillcolor=f'rgba(79, 70, 229, 0.1)',
            hovertemplate='<b>%{x}</b><br>%{y:.2%}<extra></extra>'
        )
        
        fig.update_layout(
            template=ChartConfig.get_layout_template(),
            xaxis_title=kwargs.get('labels', {}).get(x_col, x_col),
            yaxis_title=kwargs.get('labels', {}).get(y_col, y_col),
            yaxis_tickformat='.0%'
        )
        
        return fig
    
    @staticmethod
    def create_pie_chart(values, names, title, **kwargs):
        """Create styled pie chart for error distribution"""
        fig = px.pie(
            values=values,
            names=names,
            title=title,
            color_discrete_sequence=ChartConfig.COLOR_SCALES['skills'],
            **kwargs
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>%{value} 次<br>%{percent}<extra></extra>',
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        
        fig.update_layout(
            template=ChartConfig.get_layout_template(),
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_gauge_chart(value, title, max_value=1.0, **kwargs):
        """Create styled gauge chart for performance metrics"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {
                    'range': [None, max_value],
                    'tickformat': '.0%',
                    'tickcolor': ChartConfig.TEXT_SECONDARY
                },
                'bar': {
                    'color': ChartConfig.PRIMARY,
                    'thickness': 0.5
                },
                'steps': [
                    {'range': [0, 0.5], 'color': ChartConfig.ERROR},
                    {'range': [0.5, 0.7], 'color': ChartConfig.WARNING},
                    {'range': [0.7, 0.85], 'color': '#FBBF24'},
                    {'range': [0.85, 1.0], 'color': ChartConfig.SUCCESS}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.7
                }
            }
        ))
        
        fig.update_layout(
            template=ChartConfig.get_layout_template(),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    @staticmethod
    def create_heatmap(data, title, **kwargs):
        """Create styled heatmap for knowledge state visualization"""
        fig = px.imshow(
            data,
            title=title,
            color_continuous_scale=ChartConfig.COLOR_SCALES['mastery'],
            **kwargs
        )
        
        fig.update_layout(
            template=ChartConfig.get_layout_template()
        )
        
        return fig
    
    @staticmethod
    def create_scatter_chart(data, x_col, y_col, title, color_col=None, size_col=None, **kwargs):
        """Create styled scatter chart for correlation analysis"""
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            title=title,
            color=color_col,
            size=size_col,
            color_continuous_scale=ChartConfig.COLOR_SCALES['mastery'],
            **kwargs
        )
        
        fig.update_traces(
            marker=dict(
                line=dict(color=ChartConfig.PRIMARY, width=1)
            ),
            hovertemplate='<b>%{x}</b><br>%{y:.2%}<extra></extra>'
        )
        
        fig.update_layout(
            template=ChartConfig.get_layout_template(),
            xaxis_title=kwargs.get('labels', {}).get(x_col, x_col),
            yaxis_title=kwargs.get('labels', {}).get(y_col, y_col),
            yaxis_tickformat='.0%'
        )
        
        return fig
    
    @staticmethod
    def create_radar_chart(data, r_col, theta_col, title, **kwargs):
        """Create styled radar chart for multi-dimensional skill assessment"""
        fig = px.line_polar(
            data,
            r=r_col,
            theta=theta_col,
            title=title,
            line_close=True,
            **kwargs
        )
        
        fig.update_traces(
            fill='toself',
            fillcolor=f'rgba(79, 70, 229, 0.2)',
            line=dict(color=ChartConfig.PRIMARY, width=3)
        )
        
        fig.update_layout(
            template=ChartConfig.get_layout_template(),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.0%'
                )
            )
        )
        
        return fig
    
    @staticmethod
    def create_multi_line_chart(data_dict, title, **kwargs):
        """Create multi-line chart for comparing multiple students or skills"""
        fig = go.Figure()
        
        colors = ChartConfig.COLOR_SCALES['skills']
        
        for i, (name, data) in enumerate(data_dict.items()):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data.values,
                name=name,
                line=dict(color=colors[i % len(colors)], width=3),
                hovertemplate=f'<b>{name}</b><br>%{{x}}<br>%{{y:.2%}}<extra></extra>'
            ))
        
        fig.update_layout(
            template=ChartConfig.get_layout_template(),
            title=title,
            xaxis_title=kwargs.get('x_title', '时间'),
            yaxis_title=kwargs.get('y_title', '正确率'),
            yaxis_tickformat='.0%',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_funnel_chart(data, title, **kwargs):
        """Create styled funnel chart for learning path visualization"""
        fig = px.funnel(
            data,
            title=title,
            color_discrete_sequence=ChartConfig.COLOR_SCALES['progress'],
            **kwargs
        )
        
        fig.update_layout(
            template=ChartConfig.get_layout_template()
        )
        
        return fig
    
    @staticmethod
    def create_treemap(data, path, values, title, **kwargs):
        """Create styled treemap for hierarchical skill visualization"""
        fig = px.treemap(
            data,
            path=path,
            values=values,
            title=title,
            color_discrete_sequence=ChartConfig.COLOR_SCALES['skills'],
            **kwargs
        )
        
        fig.update_layout(
            template=ChartConfig.get_layout_template()
        )
        
        return fig
