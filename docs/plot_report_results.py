import pandas as pd
import os
from plotnine import *
import numpy as np
from fire import Fire
from mistletoe import Document, HTMLRenderer


def plot(report_md: str = 'report.md'):
    """Plots the report ( all markdown tables )

    :report_md:str: TODO
    :returns: TODO

    """
    report_fname = os.path.splitext(os.path.basename(report_md))[0]

    with open(report_md, 'r') as fin:
        with HTMLRenderer() as renderer:
            rendered = renderer.render(Document(fin))
            tables = pd.read_html(rendered)
            for table_id, table in enumerate(tables):
                table = pd.melt(table, id_vars=['Model', 'Pooling', 'Feature'], value_vars=[
                    "F1", "Pre","Rec", "MAE", "RMSE"])
                table['variable'] = pd.Categorical(table['variable'], categories=[
                    'F1', 'Pre',"Rec", 'MAE', 'RMSE'])
                table['Model'] = table['Model'].astype('category')
                dodge_text = position_dodge(width=0.9)
                plt = (ggplot(table, aes(x='Pooling', y='value', fill='Model'))
                       + geom_col(position='dodge')
                       + labs(x="", y="")
                       + geom_text(aes(label='value'), position=dodge_text,
                                   color='black',  format_string='{0:.1f}', va='top', size=8)
                       + theme(figure_size=(11, 6), panel_spacing=.4,)
                       + theme_dark()
                       + facet_grid(facets='variable ~ Feature',
                                    scales="free_y")

                       )
                output_filename = "{}_table{}.png".format(
                    report_fname, table_id)
                plt.save(output_filename, dpi=150)
                print("Saving to {}".format(output_filename))


if __name__ == "__main__":
    Fire(plot)
