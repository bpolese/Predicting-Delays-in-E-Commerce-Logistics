import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="E-Commerce Delivery Performance Analytics",
    page_icon="📦",
    layout="wide",
)

DATA_FILES = {
    "customers": "customers_dataset.csv",
    "orders": "orders_dataset.csv",
    "order_items": "order_items_dataset.csv",
    "sellers": "sellers_dataset.csv",
}


# -----------------------------
# Data loading and preparation
# -----------------------------

@st.cache_data(show_spinner=False)
def load_csv(file_name: str) -> pd.DataFrame:
    if not os.path.exists(file_name):
        st.error(f"Missing required file: {file_name}")
        st.stop()
    return pd.read_csv(file_name)


@st.cache_data(show_spinner=True)
def prepare_data() -> pd.DataFrame:
    customers = load_csv(DATA_FILES["customers"])
    orders = load_csv(DATA_FILES["orders"])
    order_items = load_csv(DATA_FILES["order_items"])
    sellers = load_csv(DATA_FILES["sellers"])

    required_columns = {
        "customers": ["customer_id", "customer_unique_id", "customer_city", "customer_state"],
        "orders": [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        "order_items": [
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "shipping_limit_date",
            "price",
            "freight_value",
        ],
        "sellers": ["seller_id", "seller_city", "seller_state"],
    }

    datasets = {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "sellers": sellers,
    }

    for name, cols in required_columns.items():
        missing = [c for c in cols if c not in datasets[name].columns]
        if missing:
            st.error(f"{name} is missing required columns: {missing}")
            st.stop()

    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    order_items["shipping_limit_date"] = pd.to_datetime(
        order_items["shipping_limit_date"], errors="coerce"
    )

    item_agg = (
        order_items.groupby("order_id")
        .agg(
            item_count=("order_item_id", "count"),
            unique_products=("product_id", "nunique"),
            seller_count=("seller_id", "nunique"),
            total_price=("price", "sum"),
            avg_item_price=("price", "mean"),
            total_freight=("freight_value", "sum"),
            avg_freight=("freight_value", "mean"),
            max_shipping_limit_date=("shipping_limit_date", "max"),
        )
        .reset_index()
    )

    primary_seller = (
        order_items.groupby(["order_id", "seller_id"], as_index=False)
        .agg(seller_order_value=("price", "sum"))
        .sort_values(["order_id", "seller_order_value"], ascending=[True, False])
        .drop_duplicates("order_id")
        .drop(columns=["seller_order_value"])
    )

    primary_seller = primary_seller.merge(sellers, on="seller_id", how="left")

    df = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(item_agg, on="order_id", how="left")
        .merge(primary_seller, on="order_id", how="left")
    )

    delivered = df[
        (df["order_status"] == "delivered")
        & df["order_delivered_customer_date"].notna()
        & df["order_estimated_delivery_date"].notna()
        & df["order_purchase_timestamp"].notna()
    ].copy()

    delivered["delivery_days"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    delivered["estimated_delivery_days"] = (
        delivered["order_estimated_delivery_date"]
        - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    delivered["approval_hours"] = (
        delivered["order_approved_at"] - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600

    delivered["carrier_handoff_days"] = (
        delivered["order_delivered_carrier_date"] - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    delivered["shipping_limit_days"] = (
        delivered["max_shipping_limit_date"] - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    delivered["delivery_delay_days"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    delivered["late_delivery"] = (delivered["delivery_delay_days"] > 0).astype(int)
    delivered["on_time_delivery"] = 1 - delivered["late_delivery"]

    delivered["order_month"] = delivered["order_purchase_timestamp"].dt.to_period("M").astype(str)
    delivered["purchase_month_num"] = delivered["order_purchase_timestamp"].dt.month
    delivered["purchase_weekday"] = delivered["order_purchase_timestamp"].dt.day_name()
    delivered["purchase_hour"] = delivered["order_purchase_timestamp"].dt.hour

    delivered["same_state"] = (
        delivered["customer_state"].fillna("Unknown")
        == delivered["seller_state"].fillna("Unknown")
    ).astype(int)

    delivered["freight_share"] = np.where(
        delivered["total_price"] > 0,
        delivered["total_freight"] / delivered["total_price"],
        np.nan,
    )

    delivered["order_value"] = delivered["total_price"] + delivered["total_freight"]

    numeric_cols = [
        "item_count",
        "unique_products",
        "seller_count",
        "total_price",
        "avg_item_price",
        "total_freight",
        "avg_freight",
        "order_value",
        "approval_hours",
        "shipping_limit_days",
        "estimated_delivery_days",
        "freight_share",
        "delivery_days",
        "delivery_delay_days",
    ]

    for col in numeric_cols:
        delivered[col] = pd.to_numeric(delivered[col], errors="coerce")
        delivered.loc[delivered[col] < 0, col] = np.nan

    text_cols = ["customer_city", "customer_state", "seller_city", "seller_state"]
    for col in text_cols:
        delivered[col] = (
            delivered[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.title()
        )

    delivered["customer_state"] = delivered["customer_state"].str.upper()
    delivered["seller_state"] = delivered["seller_state"].str.upper()

    delivered = delivered.dropna(
        subset=[
            "late_delivery",
            "delivery_days",
            "estimated_delivery_days",
            "total_price",
            "total_freight",
            "customer_state",
            "seller_state",
        ]
    )

    return delivered


@st.cache_resource(show_spinner=True)
def train_model(df: pd.DataFrame):
    model_df = df.copy()

    features = [
        "item_count",
        "unique_products",
        "seller_count",
        "total_price",
        "total_freight",
        "order_value",
        "avg_item_price",
        "avg_freight",
        "approval_hours",
        "shipping_limit_days",
        "estimated_delivery_days",
        "freight_share",
        "purchase_month_num",
        "purchase_hour",
        "same_state",
        "customer_state",
        "seller_state",
        "purchase_weekday",
    ]

    target = "late_delivery"

    model_df = model_df[features + [target]].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)

    X = model_df[features]
    y = model_df[target].astype(int)

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=25,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )

    stratify = y if y.nunique() == 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["On Time", "Late"],
            output_dict=True,
            zero_division=0,
        ),
    }

    importances = pipeline.named_steps["model"].feature_importances_
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .head(15)
    )

    importance_df["feature"] = (
        importance_df["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ")
        .str.title()
    )

    return pipeline, metrics, importance_df, features


# -----------------------------
# Business helper functions
# -----------------------------

def format_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.1%}"


def format_days(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.1f} days"


def risk_label(prob):
    if prob >= 0.50:
        return "High Risk"
    if prob >= 0.25:
        return "Moderate Risk"
    return "Low Risk"


def filter_data(df):
    with st.sidebar:
        st.header("Filters")

        min_date = df["order_purchase_timestamp"].min().date()
        max_date = df["order_purchase_timestamp"].max().date()

        date_range = st.date_input(
            "Order purchase date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        states = sorted(df["customer_state"].dropna().unique())
        selected_states = st.multiselect(
            "Customer states",
            options=states,
            default=states,
        )

        seller_states = sorted(df["seller_state"].dropna().unique())
        selected_seller_states = st.multiselect(
            "Seller states",
            options=seller_states,
            default=seller_states,
        )

        min_order_value = float(np.nanpercentile(df["order_value"], 1))
        max_order_value = float(np.nanpercentile(df["order_value"], 99))

        value_range = st.slider(
            "Order value range",
            min_value=0.0,
            max_value=max_order_value,
            value=(0.0, max_order_value),
            step=10.0,
        )

    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    filtered = df[
        (df["order_purchase_timestamp"].dt.date >= start_date)
        & (df["order_purchase_timestamp"].dt.date <= end_date)
        & (df["customer_state"].isin(selected_states))
        & (df["seller_state"].isin(selected_seller_states))
        & (df["order_value"].between(value_range[0], value_range[1]))
    ].copy()

    return filtered


def metric_card(label, value, help_text=None):
    st.metric(label=label, value=value, help=help_text)


def business_takeaway_box(title, bullets):
    st.markdown(f"#### {title}")
    for bullet in bullets:
        st.markdown(f"- {bullet}")


# -----------------------------
# App
# -----------------------------

df = prepare_data()
model, model_metrics, importance_df, model_features = train_model(df)

st.title("📦 E-Commerce Delivery Performance Analytics")
st.caption(
    "Executive analytics app for identifying late-delivery risk, operational bottlenecks, seller performance, and geographic patterns."
)

filtered_df = filter_data(df)

if filtered_df.empty:
    st.warning("No records match the current filters. Adjust the sidebar filters to continue.")
    st.stop()

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Delivery Risk Model",
        "Seller & Geography Diagnostics",
        "Order Risk Simulator",
        "Data Quality",
    ],
)

# -----------------------------
# Executive Overview
# -----------------------------

if page == "Executive Overview":
    st.subheader("Executive Overview")

    total_orders = filtered_df["order_id"].nunique()
    late_rate = filtered_df["late_delivery"].mean()
    avg_delivery_days = filtered_df["delivery_days"].mean()
    avg_delay_days = filtered_df.loc[filtered_df["late_delivery"] == 1, "delivery_delay_days"].mean()
    avg_order_value = filtered_df["order_value"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Delivered Orders", f"{total_orders:,.0f}")
    with c2:
        metric_card("Late Delivery Rate", format_pct(late_rate))
    with c3:
        metric_card("Avg. Delivery Time", format_days(avg_delivery_days))
    with c4:
        metric_card("Avg. Delay When Late", format_days(avg_delay_days))
    with c5:
        metric_card("Avg. Order Value", f"${avg_order_value:,.2f}")

    st.divider()

    monthly = (
        filtered_df.groupby("order_month", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            late_rate=("late_delivery", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
        )
        .sort_values("order_month")
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            monthly,
            x="order_month",
            y="late_rate",
            markers=True,
            title="Late Delivery Rate Over Time",
            labels={
                "order_month": "Order Month",
                "late_rate": "Late Delivery Rate",
            },
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            monthly,
            x="order_month",
            y="orders",
            title="Delivered Order Volume Over Time",
            labels={
                "order_month": "Order Month",
                "orders": "Delivered Orders",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    by_state = (
        filtered_df.groupby("customer_state", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            late_rate=("late_delivery", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
        )
        .query("orders >= 50")
        .sort_values("late_rate", ascending=False)
        .head(10)
    )

    fig = px.bar(
        by_state,
        x="late_rate",
        y="customer_state",
        orientation="h",
        title="Highest Late Delivery Rates by Customer State",
        labels={
            "late_rate": "Late Delivery Rate",
            "customer_state": "Customer State",
        },
        text=by_state["late_rate"].map(lambda x: f"{x:.1%}"),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    worst_state = by_state.iloc[0]["customer_state"] if not by_state.empty else "N/A"
    worst_state_rate = by_state.iloc[0]["late_rate"] if not by_state.empty else np.nan

    business_takeaway_box(
        "Key Takeaways",
        [
            f"The selected data contains {total_orders:,.0f} delivered orders, with a late delivery rate of {format_pct(late_rate)}.",
            f"The average delivered order takes {format_days(avg_delivery_days)} from purchase to customer delivery.",
            f"The highest-risk customer state in the current filter is {worst_state}, with a late rate of {format_pct(worst_state_rate)}.",
            "Operational focus should start with seller-state/customer-state lanes that combine high volume with high late-delivery rates.",
        ],
    )


# -----------------------------
# Delivery Risk Model
# -----------------------------

elif page == "Delivery Risk Model":
    st.subheader("Delivery Risk Model")

    st.markdown(
        "This model predicts whether an order is likely to be delivered late using order value, freight, seller/customer geography, purchase timing, and fulfillment features."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Accuracy", format_pct(model_metrics["accuracy"]))
    with c2:
        metric_card("Precision", format_pct(model_metrics["precision"]))
    with c3:
        metric_card("Recall", format_pct(model_metrics["recall"]))
    with c4:
        metric_card("ROC AUC", f"{model_metrics['roc_auc']:.3f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        cm = model_metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual On Time", "Actual Late"],
            columns=["Predicted On Time", "Predicted Late"],
        )
        fig = px.imshow(
            cm_df,
            text_auto=True,
            title="Model Confusion Matrix",
            labels=dict(color="Orders"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            importance_df.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="Top Model Drivers of Late Delivery Risk",
            labels={
                "importance": "Relative Importance",
                "feature": "Feature",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    report = pd.DataFrame(model_metrics["classification_report"]).T
    report = report.loc[["On Time", "Late"], ["precision", "recall", "f1-score", "support"]]
    report["precision"] = report["precision"].map(lambda x: f"{x:.1%}")
    report["recall"] = report["recall"].map(lambda x: f"{x:.1%}")
    report["f1-score"] = report["f1-score"].map(lambda x: f"{x:.1%}")
    report["support"] = report["support"].map(lambda x: f"{x:,.0f}")

    st.markdown("#### Classification Summary")
    st.dataframe(report, use_container_width=True)

    business_takeaway_box(
        "How to Use This Model",
        [
            "Use the model as an early warning system for orders that may need proactive monitoring.",
            "Prioritize orders with high freight burden, longer promised delivery windows, high order complexity, or risky geography lanes.",
            "The model supports operational triage; it should not replace business judgment or seller-level investigation.",
        ],
    )


# -----------------------------
# Seller & Geography Diagnostics
# -----------------------------

elif page == "Seller & Geography Diagnostics":
    st.subheader("Seller & Geography Diagnostics")

    min_orders = st.slider(
        "Minimum delivered orders per seller/state lane",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
    )

    seller_perf = (
        filtered_df.groupby(["seller_id", "seller_city", "seller_state"], as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            late_rate=("late_delivery", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
            avg_delay_when_late=("delivery_delay_days", lambda x: x[x > 0].mean()),
            avg_order_value=("order_value", "mean"),
            total_order_value=("order_value", "sum"),
        )
    )

    seller_perf = seller_perf[seller_perf["orders"] >= min_orders].copy()

    high_risk_sellers = seller_perf.sort_values(
        ["late_rate", "orders"], ascending=[False, False]
    ).head(15)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            high_risk_sellers.sort_values("late_rate", ascending=True),
            x="late_rate",
            y="seller_id",
            orientation="h",
            title="Highest-Risk Sellers by Late Delivery Rate",
            labels={
                "late_rate": "Late Delivery Rate",
                "seller_id": "Seller ID",
            },
            hover_data=["seller_city", "seller_state", "orders", "avg_delivery_days"],
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        lane_perf = (
            filtered_df.groupby(["seller_state", "customer_state"], as_index=False)
            .agg(
                orders=("order_id", "nunique"),
                late_rate=("late_delivery", "mean"),
                avg_delivery_days=("delivery_days", "mean"),
            )
        )
        lane_perf = lane_perf[lane_perf["orders"] >= min_orders].copy()
        lane_perf["lane"] = lane_perf["seller_state"] + " → " + lane_perf["customer_state"]

        risky_lanes = lane_perf.sort_values(
            ["late_rate", "orders"], ascending=[False, False]
        ).head(15)

        fig = px.bar(
            risky_lanes.sort_values("late_rate", ascending=True),
            x="late_rate",
            y="lane",
            orientation="h",
            title="Highest-Risk Seller-to-Customer State Lanes",
            labels={
                "late_rate": "Late Delivery Rate",
                "lane": "Seller → Customer State",
            },
            hover_data=["orders", "avg_delivery_days"],
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Seller Performance Table")
    seller_table = seller_perf.sort_values(
        ["late_rate", "orders"], ascending=[False, False]
    ).copy()

    seller_table["late_rate"] = seller_table["late_rate"].map(lambda x: f"{x:.1%}")
    seller_table["avg_delivery_days"] = seller_table["avg_delivery_days"].map(lambda x: f"{x:.1f}")
    seller_table["avg_delay_when_late"] = seller_table["avg_delay_when_late"].map(
        lambda x: "N/A" if pd.isna(x) else f"{x:.1f}"
    )
    seller_table["avg_order_value"] = seller_table["avg_order_value"].map(lambda x: f"${x:,.2f}")
    seller_table["total_order_value"] = seller_table["total_order_value"].map(lambda x: f"${x:,.2f}")

    st.dataframe(
        seller_table[
            [
                "seller_id",
                "seller_city",
                "seller_state",
                "orders",
                "late_rate",
                "avg_delivery_days",
                "avg_delay_when_late",
                "avg_order_value",
                "total_order_value",
            ]
        ],
        use_container_width=True,
        height=450,
    )

    same_state_summary = (
        filtered_df.groupby("same_state", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            late_rate=("late_delivery", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
        )
    )

    same_state_summary["route_type"] = np.where(
        same_state_summary["same_state"] == 1,
        "Seller and customer in same state",
        "Seller and customer in different states",
    )

    fig = px.bar(
        same_state_summary,
        x="route_type",
        y="late_rate",
        title="Late Delivery Rate by Same-State vs. Cross-State Fulfillment",
        labels={
            "route_type": "Fulfillment Type",
            "late_rate": "Late Delivery Rate",
        },
        text=same_state_summary["late_rate"].map(lambda x: f"{x:.1%}"),
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    business_takeaway_box(
        "Operational Recommendations",
        [
            "Review high-volume sellers with above-average late rates before low-volume outliers.",
            "Investigate state-to-state lanes where late delivery is high and order volume is meaningful.",
            "Compare same-state and cross-state fulfillment to understand whether distance-related routing is driving service failures.",
        ],
    )


# -----------------------------
# Order Risk Simulator
# -----------------------------

elif page == "Order Risk Simulator":
    st.subheader("Order Risk Simulator")

    st.markdown(
        "Enter a hypothetical order profile to estimate late-delivery risk before the order is completed."
    )

    states = sorted(df["customer_state"].dropna().unique())
    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    col1, col2, col3 = st.columns(3)

    with col1:
        item_count = st.number_input("Item count", min_value=1, max_value=25, value=1)
        unique_products = st.number_input("Unique products", min_value=1, max_value=25, value=1)
        seller_count = st.number_input("Seller count", min_value=1, max_value=10, value=1)
        total_price = st.number_input("Product value", min_value=0.0, value=120.0, step=10.0)

    with col2:
        total_freight = st.number_input("Freight value", min_value=0.0, value=20.0, step=5.0)
        approval_hours = st.number_input("Approval time in hours", min_value=0.0, value=2.0, step=1.0)
        shipping_limit_days = st.number_input("Shipping limit days", min_value=0.0, value=4.0, step=1.0)
        estimated_delivery_days = st.number_input("Promised delivery window in days", min_value=1.0, value=12.0, step=1.0)

    with col3:
        customer_state = st.selectbox("Customer state", states, index=states.index("SP") if "SP" in states else 0)
        seller_state = st.selectbox("Seller state", states, index=states.index("SP") if "SP" in states else 0)
        purchase_weekday = st.selectbox("Purchase weekday", weekdays)
        purchase_hour = st.slider("Purchase hour", min_value=0, max_value=23, value=12)

    order_value = total_price + total_freight
    avg_item_price = total_price / max(item_count, 1)
    avg_freight = total_freight / max(item_count, 1)
    freight_share = total_freight / total_price if total_price > 0 else 0
    same_state = int(customer_state == seller_state)

    input_df = pd.DataFrame(
        [
            {
                "item_count": item_count,
                "unique_products": unique_products,
                "seller_count": seller_count,
                "total_price": total_price,
                "total_freight": total_freight,
                "order_value": order_value,
                "avg_item_price": avg_item_price,
                "avg_freight": avg_freight,
                "approval_hours": approval_hours,
                "shipping_limit_days": shipping_limit_days,
                "estimated_delivery_days": estimated_delivery_days,
                "freight_share": freight_share,
                "purchase_month_num": datetime.today().month,
                "purchase_hour": purchase_hour,
                "same_state": same_state,
                "customer_state": customer_state,
                "seller_state": seller_state,
                "purchase_weekday": purchase_weekday,
            }
        ]
    )

    risk_prob = model.predict_proba(input_df)[0, 1]
    label = risk_label(risk_prob)

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Estimated Late Delivery Risk", format_pct(risk_prob))
    with c2:
        metric_card("Risk Category", label)
    with c3:
        metric_card("Order Value", f"${order_value:,.2f}")

    if risk_prob >= 0.50:
        st.error(
            "This order has elevated late-delivery risk. Consider proactive monitoring, seller follow-up, or customer communication."
        )
    elif risk_prob >= 0.25:
        st.warning(
            "This order has moderate risk. Monitor fulfillment timing and watch for seller or carrier handoff delays."
        )
    else:
        st.success(
            "This order has relatively low predicted late-delivery risk based on historical patterns."
        )

    st.markdown("#### Scenario Summary")
    scenario_summary = pd.DataFrame(
        {
            "Input": [
                "Item count",
                "Seller count",
                "Product value",
                "Freight value",
                "Freight share",
                "Customer state",
                "Seller state",
                "Same-state fulfillment",
                "Promised delivery window",
                "Approval time",
            ],
            "Value": [
                item_count,
                seller_count,
                f"${total_price:,.2f}",
                f"${total_freight:,.2f}",
                f"{freight_share:.1%}",
                customer_state,
                seller_state,
                "Yes" if same_state else "No",
                f"{estimated_delivery_days:.1f} days",
                f"{approval_hours:.1f} hours",
            ],
        }
    )
    st.dataframe(scenario_summary, use_container_width=True, hide_index=True)


# -----------------------------
# Data Quality
# -----------------------------

elif page == "Data Quality":
    st.subheader("Data Quality & Processing Summary")

    raw_counts = {
        "Prepared delivered orders": len(df),
        "Unique orders": df["order_id"].nunique(),
        "Unique customers": df["customer_unique_id"].nunique(),
        "Unique sellers": df["seller_id"].nunique(),
        "Customer states": df["customer_state"].nunique(),
        "Seller states": df["seller_state"].nunique(),
    }

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Prepared Rows", f"{raw_counts['Prepared delivered orders']:,.0f}")
    with c2:
        metric_card("Unique Customers", f"{raw_counts['Unique customers']:,.0f}")
    with c3:
        metric_card("Unique Sellers", f"{raw_counts['Unique sellers']:,.0f}")

    st.markdown("#### Data Preparation Rules Applied")
    st.markdown(
        """
- Used delivered orders with valid purchase, estimated delivery, and actual delivery dates.
- Aggregated item-level records to the order level.
- Assigned each order to its primary seller by largest item value.
- Created delivery timing, freight, order complexity, geography, and late-delivery features.
- Removed invalid negative timing values from model-ready numeric fields.
"""
    )

    missing_summary = (
        df.isna()
        .mean()
        .reset_index()
        .rename(columns={"index": "Column", 0: "Missing Share"})
        .sort_values("Missing Share", ascending=False)
    )

    missing_summary["Missing Share"] = missing_summary["Missing Share"].map(lambda x: f"{x:.1%}")

    st.markdown("#### Missing Value Summary After Preparation")
    st.dataframe(missing_summary, use_container_width=True, height=450)

    st.markdown("#### Prepared Data Preview")
    preview_cols = [
        "order_id",
        "customer_state",
        "seller_state",
        "item_count",
        "total_price",
        "total_freight",
        "order_value",
        "delivery_days",
        "estimated_delivery_days",
        "delivery_delay_days",
        "late_delivery",
    ]
    st.dataframe(df[preview_cols].head(100), use_container_width=True)
