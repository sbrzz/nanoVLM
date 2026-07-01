import base64
import io
import argparse

import dash
from dash import dcc, html, Input, Output, State, dash_table
from datasets import load_dataset
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("hf_dataset", type=str)
args = parser.parse_args()

DATASET_NAME = args.hf_dataset
SPLIT = "train"
PAGE_SIZE = 12
THUMB_SIZE = (256, 256)

dataset = load_dataset(DATASET_NAME, split=SPLIT)
N = len(dataset)
COLUMNS = dataset.column_names
IMAGE_COLS = [c for c in COLUMNS if str(dataset.features[c]).startswith("Image")]
TEXT_COLS = [c for c in COLUMNS if c not in IMAGE_COLS]


def img_to_b64(img: Image.Image) -> str:
    img = img.copy()
    img.thumbnail(THUMB_SIZE)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_page(page: int, search: str = ""):
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, N)

    # HF datasets support fast batched slicing — pull only this page's rows
    batch = dataset[start:end]  # dict of column -> list, no full-dataset materialization

    cards = []
    for i in range(end - start):
        idx = start + i
        row_text = " ".join(str(batch[c][i]) for c in TEXT_COLS).lower()
        if search and search.lower() not in row_text:
            continue

        img_el = None
        if IMAGE_COLS:
            img_el = html.Img(
                src=img_to_b64(batch[IMAGE_COLS[0]][i]),
                style={"width": "100%", "borderRadius": "6px"},
            )

        text_rows = [
            html.Div(
                [html.Span(f"{c}: ", style={"fontWeight": "bold"}), str(batch[c][i])[:200]],
                style={"fontSize": "12px", "marginTop": "4px"},
            )
            for c in TEXT_COLS
        ]

        cards.append(
            html.Div(
                [img_el, html.Div(f"#{idx}", style={"fontSize": "11px", "color": "#888"}), *text_rows],
                style={
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                    "padding": "8px",
                    "width": "220px",
                },
            )
        )

    return cards


app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H3(f"{DATASET_NAME} ({N} rows)"),
        dcc.Input(id="search", type="text", placeholder="Filter text columns...", debounce=True,
                   style={"width": "300px", "marginRight": "12px"}),
        html.Button("Prev", id="prev", n_clicks=0),
        html.Button("Next", id="next", n_clicks=0, style={"marginLeft": "8px"}),
        html.Span(id="page-label", style={"marginLeft": "12px"}),
        dcc.Store(id="page", data=0),
        html.Div(
            id="grid",
            style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginTop": "16px"},
        ),
    ],
    style={"padding": "20px", "fontFamily": "sans-serif"},
)


@app.callback(
    Output("page", "data"),
    Input("prev", "n_clicks"),
    Input("next", "n_clicks"),
    State("page", "data"),
    prevent_initial_call=True,
)
def change_page(prev_clicks, next_clicks, page):
    trigger = dash.callback_context.triggered_id
    max_page = (N - 1) // PAGE_SIZE
    if trigger == "next":
        return min(page + 1, max_page)
    if trigger == "prev":
        return max(page - 1, 0)
    return page


@app.callback(
    Output("grid", "children"),
    Output("page-label", "children"),
    Input("page", "data"),
    Input("search", "value"),
)
def update_grid(page, search):
    max_page = (N - 1) // PAGE_SIZE
    label = f"Page {page + 1} / {max_page + 1}"
    return render_page(page, search or ""), label


if __name__ == "__main__":
    app.run(debug=True)